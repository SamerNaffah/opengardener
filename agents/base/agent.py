"""
EmergentAgent — base class for all OpenGardener agents.

Lifecycle:
  1. before_task(description) → consults Soil, returns strategy dict
  2. [agent subclass executes the task]
  3. after_task(description, success, resources) → deposits pheromone trail
"""

import logging
import random
import time
import uuid
from typing import Optional

from sentence_transformers import SentenceTransformer

from .soil_client import SoilClient, SoilQueryResult
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# Lazy-loaded singleton — shared across all agents in the same process
_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedder loaded.")
    return _embedder


class TaskOutcome:
    def __init__(self, success: bool, data=None, error: str = ""):
        self.success = success
        self.data = data
        self.error = error


class EmergentAgent:
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

        logger.info(f"Agent {self.id} initialized (type={self.__class__.__name__})")

    @property
    def task_domain(self) -> str:
        """Override in subclasses to return the domain name."""
        return self.genome.get("specialization", "general")

    @property
    def failure_rate(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.failures / self.tasks_completed

    def embed(self, text: str) -> list[float]:
        return self.embedder.encode(text).tolist()

    def before_task(self, task_description: str) -> dict:
        """
        Consult the Soil before executing a task.
        Returns the strategy to use (exploit known good approaches or explore new ones).
        """
        embedding = self.embed(task_description)

        # Query the soil for similar successful tasks
        similar = self.soil.query_similar(
            embedding=embedding,
            limit=5,
            threshold=0.7,
            domain=self.task_domain,
        )

        # Filter to successes only
        successes = [r for r in similar if r.outcome == "success"]

        if successes:
            # EXPLOIT: adopt the most successful known approach
            best = max(successes, key=lambda r: r.success_rate)
            self.current_strategy = best.approach
            self.current_confidence = best.success_rate
            logger.info(
                f"[{self.id}] EXPLOIT: using trail {best.trail_id[:8]}... "
                f"(similarity={best.success_rate:.2f}, hits={best.hits})"
            )
        else:
            # EXPLORE: generate a novel approach
            self.current_confidence = 0.3

            # Check for failure markers to avoid
            failures = [r for r in similar if r.outcome == "failure"]
            failed_approaches = [r.approach for r in failures]

            # Try LLM-guided mutation first
            if self.llm.is_available() and self.current_confidence < 0.5:
                llm_approach = self.llm.suggest_approach(task_description, [
                    str(f) for f in failed_approaches[:3]
                ])
                if llm_approach:
                    self.current_strategy = llm_approach
                    logger.info(f"[{self.id}] EXPLORE (LLM): generated new approach")
                    return self.current_strategy

            # Fall back to rule-based mutation
            self.current_strategy = self.mutate_approach(task_description)
            logger.info(f"[{self.id}] EXPLORE (rule-based): mutated approach")

        return self.current_strategy

    def after_task(
        self,
        task_description: str,
        outcome: TaskOutcome,
        resources: Optional[dict] = None,
    ):
        """
        Deposit a pheromone trail after completing (or failing) a task.
        """
        resources = resources or {}
        embedding = self.embed(task_description)

        if outcome.success:
            self.tasks_completed += 1
            self.soil.leave_trail(
                embedding=embedding,
                outcome="success",
                approach=self.current_strategy,
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
                failed_approach=self.current_strategy,
                error=outcome.error,
                agent_id=self.id,
                severity="medium",
                task_domain=self.task_domain,
                avoid_in_future=True,
            )
            logger.info(f"[{self.id}] Task FAILURE — marker deposited: {outcome.error}")

        # Accumulate experience locally
        self.experience_buffer.append({
            "task": task_description,
            "approach": self.current_strategy,
            "success": outcome.success,
            "resources": resources,
        })

        # Periodically consolidate local learning
        if len(self.experience_buffer) >= 100:
            self._consolidate_learning()

        # Report health to Gardener every 10 tasks
        if self.tasks_completed % 10 == 0:
            self._report_health()

    def mutate_approach(self, task_description: str) -> dict:
        """
        Rule-based approach mutation.
        - No history → use genome default
        - Has history → mutate last success by varying one numeric parameter ±20%
        """
        if not self.experience_buffer:
            return self.genome.get("default_approach", {"method": "default"})

        successes = [e for e in self.experience_buffer if e["success"]]
        if not successes:
            return self.genome.get("fallback_approach", {"method": "fallback"})

        base = dict(successes[-1]["approach"])

        # Randomly vary one numeric parameter
        numeric_keys = [k for k, v in base.items() if isinstance(v, (int, float))]
        if numeric_keys:
            key = random.choice(numeric_keys)
            base[key] = base[key] * random.uniform(0.8, 1.2)

        return base

    def _consolidate_learning(self):
        """Archive successful patterns and trim experience buffer."""
        successes = [e for e in self.experience_buffer if e["success"]]
        if len(successes) > 0:
            success_rate = len(successes) / len(self.experience_buffer)
            logger.info(
                f"[{self.id}] Consolidating: {len(self.experience_buffer)} experiences, "
                f"success_rate={success_rate:.2f}"
            )

        # Keep last 50 experiences
        self.experience_buffer = self.experience_buffer[-50:]

    def _report_health(self):
        """
        Send health metrics to the Gardener.
        Returns True if the Gardener has ordered this agent to terminate.
        """
        try:
            import os
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
            from generated import gardener_pb2, gardener_pb2_grpc
            import grpc

            host = os.getenv("GARDENER_GRPC_HOST", "localhost")
            port = os.getenv("GARDENER_GRPC_PORT", "50052")

            with grpc.insecure_channel(f"{host}:{port}") as channel:
                stub = gardener_pb2_grpc.GardenerStub(channel)
                report = gardener_pb2.HealthReport(
                    agent_id=self.id,
                    failure_rate=self.failure_rate,
                    tasks_completed=self.tasks_completed,
                    cpu_ms=0.0,
                    memory_mb=0.0,
                    current_domain=self.task_domain,
                    status="active" if self.failure_rate < 0.5 else "struggling",
                )
                ack = stub.ReportHealth(report, timeout=5.0)

                if ack.should_terminate:
                    logger.warning(
                        f"[{self.id}] PRUNE SIGNAL received from Gardener: {ack.terminate_reason}. "
                        f"Archiving knowledge and shutting down."
                    )
                    self._archive_and_exit(ack.terminate_reason)
                    return True

                if ack.message and ack.message != "OK":
                    logger.info(f"[{self.id}] Gardener feedback: {ack.message}")

        except Exception as e:
            logger.debug(f"[{self.id}] Health report failed (non-critical): {e}")

        return False

    def _archive_and_exit(self, reason: str):
        """
        Graceful shutdown: notify Gardener of termination, then exit.
        Soil trails are preserved — they remain as collective memory for future agents.
        """
        try:
            import os
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
            from generated import gardener_pb2, gardener_pb2_grpc
            import grpc

            host = os.getenv("GARDENER_GRPC_HOST", "localhost")
            port = os.getenv("GARDENER_GRPC_PORT", "50052")

            with grpc.insecure_channel(f"{host}:{port}") as channel:
                stub = gardener_pb2_grpc.GardenerStub(channel)
                stub.RequestTermination(
                    gardener_pb2.TerminationRequest(
                        agent_id=self.id,
                        reason=reason,
                    ),
                    timeout=5.0,
                )
                logger.info(f"[{self.id}] Termination confirmed to Gardener.")
        except Exception as e:
            logger.debug(f"[{self.id}] Termination notification failed: {e}")
        finally:
            import os as _os
            _os._exit(0)
