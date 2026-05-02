"""
EmergentAgent — base class for all OpenGardener agents.

Lifecycle:
  1. before_task(description) → consults Soil, returns strategy dict
  2. [agent subclass executes the task]
  3. after_task(description, success, resources) → deposits pheromone trail

Key design decisions worth knowing:
  * The sentence-transformers model is a process-global singleton. Loading it
    is ~1s and ~80MB of RAM, so we never want more than one per process.
  * The Gardener gRPC channel is also a singleton — opening a fresh channel for
    every health report adds 50–200ms of TCP/TLS handshake per call.
  * Health is reported after EVERY task (cheap; <2ms over local network) so
    prune signals are honoured promptly. The previous "every 10 tasks" cadence
    meant a struggling agent could complete dozens more failed tasks before
    getting the kill signal.
  * Soil scoring uses BOTH cosine similarity AND the producing agent's historical
    success rate, mirroring the Rust-side compose_score().
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import threading
import uuid
from typing import Any, Optional

from sentence_transformers import SentenceTransformer

from .llm_client import LLMClient
from .soil_client import SoilClient, SoilQueryResult

logger = logging.getLogger(__name__)

# ─── Process-global embedder ────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_embedder_lock = threading.Lock()


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedder loaded.")
    return _embedder


class TaskOutcome:
    __slots__ = ("success", "data", "error")

    def __init__(self, success: bool, data: Any = None, error: str = ""):
        self.success = success
        self.data = data
        self.error = error


# ─── Cached gRPC stub for the Gardener ──────────────────────────────────────
_gardener_stub_cache: dict[str, Any] = {}


def _get_gardener_stub() -> Optional[Any]:
    """Return a cached `GardenerStub`. Imports lazily so unit tests that don't
    have generated stubs available can still import this module."""
    cached = _gardener_stub_cache.get("stub")
    if cached is not None:
        return cached

    # protoc-generated _pb2_grpc files use bare `import x_pb2`, so the generated/
    # dir itself (not its parent) must be on sys.path.
    gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
    if gen_dir not in sys.path:
        sys.path.insert(0, gen_dir)
    try:
        import grpc
        import gardener_pb2_grpc  # type: ignore[import-not-found]
    except ImportError as e:
        logger.warning("Gardener gRPC stubs unavailable (%s) — health reports disabled.", e)
        return None

    host = os.getenv("GARDENER_GRPC_HOST", "localhost")
    port = os.getenv("GARDENER_GRPC_PORT", "50052")
    channel = grpc.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.keepalive_time_ms", 30_000),
            ("grpc.keepalive_timeout_ms", 10_000),
            ("grpc.keepalive_permit_without_calls", 1),
        ],
    )
    stub = gardener_pb2_grpc.GardenerStub(channel)
    _gardener_stub_cache["channel"] = channel
    _gardener_stub_cache["stub"] = stub
    return stub


class EmergentAgent:
    # Tuneable: how often to push a health report (in tasks).
    # Set to 1 so prune signals propagate immediately.
    HEALTH_REPORT_EVERY = 1
    # Tuneable: at what experience-buffer depth to consolidate.
    CONSOLIDATE_AT = 100
    # Default scoring blend: best = similarity * (β + (1-β) * success_rate).
    SUCCESS_RATE_WEIGHT = 0.5

    def __init__(self, agent_id: Optional[str] = None, genome: Optional[dict] = None):
        self.id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.genome = genome or {}
        self.soil = SoilClient()
        self.llm = LLMClient()
        self.embedder = get_embedder()

        # Local state
        self.current_strategy: dict = {}
        self.current_confidence: float = 0.3
        self.experience_buffer: list[dict] = []
        self.tasks_completed: int = 0
        self.failures: int = 0
        self.terminated: bool = False

        logger.info(f"Agent {self.id} initialized (type={self.__class__.__name__})")

    # ─── Properties ─────────────────────────────────────────────────────────
    @property
    def task_domain(self) -> str:
        """Override in subclasses to return the domain name."""
        return self.genome.get("specialization", "general")

    @property
    def failure_rate(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.failures / self.tasks_completed

    # ─── Embedding helper ───────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        # show_progress_bar=False avoids cluttering logs with tqdm under Docker.
        return self.embedder.encode(text, show_progress_bar=False).tolist()

    # ─── Lifecycle hooks ────────────────────────────────────────────────────
    def before_task(self, task_description: str) -> dict:
        """
        Consult the Soil before executing a task.
        Returns the strategy to use (exploit known good approaches or explore new ones).
        """
        embedding = self.embed(task_description)

        # Query the soil for similar past tasks (both success and failure outcomes).
        similar = self.soil.query_similar(
            embedding=embedding,
            limit=5,
            threshold=0.7,
            domain=self.task_domain,
        )

        successes = [r for r in similar if r.outcome == "success"]

        if successes:
            # EXPLOIT: pick by composite (similarity × producing-agent reputation),
            # mirroring the Rust-side compose_score so client and server agree.
            best = max(successes, key=self._exploit_score)
            # `approach` is sometimes a dict, sometimes a str (LLM-suggested raw).
            self.current_strategy = self._coerce_strategy(best.approach)
            # Confidence reflects the actual producing agent's track record,
            # not the cosine similarity (which was the previous bug).
            self.current_confidence = best.success_rate or best.similarity
            logger.info(
                f"[{self.id}] EXPLOIT trail={best.trail_id[:8]}.. "
                f"sim={best.similarity:.2f} success_rate={best.success_rate:.2f} hits={best.hits}"
            )
            return self.current_strategy

        # EXPLORE — no usable success trails for this task.
        self.current_confidence = 0.3
        failures = [r for r in similar if r.outcome == "failure"]
        failed_approaches = [str(r.approach) for r in failures[:3]]

        # Try LLM-guided mutation first if available.
        if self.llm.is_available():
            llm_approach = self.llm.suggest_approach(task_description, failed_approaches)
            if llm_approach:
                self.current_strategy = self._coerce_strategy(llm_approach)
                logger.info(f"[{self.id}] EXPLORE (LLM): generated new approach")
                return self.current_strategy

        # Fall back to rule-based mutation.
        self.current_strategy = self.mutate_approach(task_description)
        logger.info(f"[{self.id}] EXPLORE (rule-based): mutated approach")
        return self.current_strategy

    def after_task(
        self,
        task_description: str,
        outcome: TaskOutcome,
        resources: Optional[dict] = None,
    ):
        """Deposit a pheromone trail after completing (or failing) a task."""
        resources = resources or {}
        embedding = self.embed(task_description)
        approach_for_wire = self._coerce_strategy(self.current_strategy)

        if outcome.success:
            self.tasks_completed += 1
            self.soil.leave_trail(
                embedding=embedding,
                outcome="success",
                approach=approach_for_wire,
                agent_id=self.id,
                task_domain=self.task_domain,
                task_summary=task_description[:200],
                resources=resources,
            )
            logger.info(f"[{self.id}] Task SUCCESS — trail deposited")
        else:
            self.tasks_completed += 1
            self.failures += 1
            self.soil.mark_failure(
                embedding=embedding,
                failed_approach=approach_for_wire,
                error=outcome.error,
                agent_id=self.id,
                severity="medium",
                task_domain=self.task_domain,
                avoid_in_future=True,
            )
            logger.info(f"[{self.id}] Task FAILURE — marker deposited: {outcome.error[:80]}")

        # Local experience buffer (used for rule-based mutation).
        self.experience_buffer.append({
            "task": task_description,
            "approach": approach_for_wire,
            "success": outcome.success,
            "resources": resources,
        })

        if len(self.experience_buffer) >= self.CONSOLIDATE_AT:
            self._consolidate_learning()

        # Health report every task — cheap and prune-signal-sensitive.
        if self.tasks_completed % self.HEALTH_REPORT_EVERY == 0:
            self._report_health()

    # ─── Mutation ───────────────────────────────────────────────────────────
    def mutate_approach(self, task_description: str) -> dict:
        """
        Rule-based approach mutation.
          - No history → use genome default.
          - Has history → vary one numeric parameter ±20%.
          - Mutation temperature decays as confidence grows (more conservative
            once we have a solid strategy).
        """
        if not self.experience_buffer:
            return dict(self.genome.get("default_approach", {"method": "default"}))

        successes = [e for e in self.experience_buffer if e["success"]]
        if not successes:
            return dict(self.genome.get("fallback_approach", {"method": "fallback"}))

        base = dict(successes[-1]["approach"])
        # Temperature shrinks with confidence: agents that already know what works
        # mutate less aggressively. Range: ±[0.10, 0.40].
        temperature = max(0.10, 0.40 * (1.0 - self.current_confidence))
        numeric_keys = [k for k, v in base.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric_keys:
            key = random.choice(numeric_keys)
            base[key] = base[key] * random.uniform(1.0 - temperature, 1.0 + temperature)
        return base

    # ─── Internal helpers ───────────────────────────────────────────────────
    def _exploit_score(self, r: SoilQueryResult) -> float:
        """Composite score for picking the best trail.

        Mirrors Rust-side compose_score so client/server are coherent.
        """
        if r.outcome != "success":
            return -1.0
        beta = self.SUCCESS_RATE_WEIGHT
        sim = max(0.0, r.similarity)
        sr = max(0.0, r.success_rate)
        # +log(1+hits) keeps well-trodden trails attractive without exploding.
        import math
        hit_bonus = math.log1p(max(0, r.hits))
        return sim * (beta + (1.0 - beta) * sr) * (1.0 + 0.1 * hit_bonus)

    @staticmethod
    def _coerce_strategy(strategy: Any) -> dict:
        """Normalise whatever shape the strategy is in into a dict.

        The LLM can return a JSON string, a dict, or a raw string. Soil-stored
        approaches are JSON strings. We always serialise as dict on the wire.
        """
        if isinstance(strategy, dict):
            return strategy
        if isinstance(strategy, str):
            try:
                parsed = json.loads(strategy)
                return parsed if isinstance(parsed, dict) else {"raw": strategy}
            except json.JSONDecodeError:
                return {"raw": strategy}
        return {"value": str(strategy)}

    def _consolidate_learning(self):
        """Archive successful patterns and trim experience buffer."""
        successes = [e for e in self.experience_buffer if e["success"]]
        if successes:
            success_rate = len(successes) / len(self.experience_buffer)
            logger.info(
                f"[{self.id}] Consolidating: {len(self.experience_buffer)} experiences, "
                f"success_rate={success_rate:.2f}"
            )
        # Keep last 50 experiences.
        self.experience_buffer = self.experience_buffer[-50:]

    # ─── Gardener health & graceful prune ──────────────────────────────────
    def _report_health(self):
        """Send health metrics to the Gardener.

        Returns True if the Gardener has ordered this agent to terminate.
        """
        stub = _get_gardener_stub()
        if stub is None:
            return False

        try:
            import gardener_pb2  # type: ignore[import-not-found]
            import grpc
        except ImportError:
            return False

        try:
            status = "struggling" if self.failure_rate > 0.5 else "active"
            report = gardener_pb2.HealthReport(
                agent_id=self.id,
                failure_rate=self.failure_rate,
                tasks_completed=self.tasks_completed,
                cpu_ms=0.0,
                memory_mb=0.0,
                current_domain=self.task_domain,
                status=status,
            )
            ack = stub.ReportHealth(report, timeout=5.0)
            if ack.should_terminate:
                logger.warning(
                    f"[{self.id}] PRUNE SIGNAL received: {ack.terminate_reason}. "
                    "Archiving knowledge and shutting down."
                )
                self._archive_and_exit(ack.terminate_reason)
                return True
            if ack.message and ack.message != "OK":
                logger.info(f"[{self.id}] Gardener feedback: {ack.message}")
        except grpc.RpcError as e:
            logger.debug(f"[{self.id}] Health report failed (non-critical): {e.details()}")
        except Exception as e:  # never let health reporting kill the loop
            logger.debug(f"[{self.id}] Health report failed (non-critical): {e}")
        return False

    def _archive_and_exit(self, reason: str):
        """
        Graceful shutdown: notify Gardener of termination, then exit cleanly.
        Soil trails are preserved — they remain as collective memory for future agents.
        """
        self.terminated = True
        try:
            stub = _get_gardener_stub()
            if stub is not None:
                import gardener_pb2  # type: ignore[import-not-found]
                stub.RequestTermination(
                    gardener_pb2.TerminationRequest(agent_id=self.id, reason=reason),
                    timeout=5.0,
                )
                logger.info(f"[{self.id}] Termination confirmed to Gardener.")
        except Exception as e:
            logger.debug(f"[{self.id}] Termination notification failed: {e}")
        finally:
            # sys.exit raises SystemExit which the run loop catches and breaks
            # cleanly out of, ensuring open files and channels get closed.
            sys.exit(0)
