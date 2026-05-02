"""
gRPC client for the Rust Soil service.

Wraps generated protobuf stubs with a clean Python interface. The channel is
opened lazily on first use and reused for the lifetime of the process. All
methods are thread-safe.

After the proto change, `SoilQueryResult` now exposes BOTH:
  * `similarity`   — cosine similarity to the query embedding (0.0–1.0)
  * `success_rate` — historical EMA success rate of the producing agent

The previous code overloaded `success_rate` with the similarity value, which
silently broke the EXPLOIT branch's "pick most-successful trail" logic.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional

# protoc-generated _pb2_grpc files use bare `import x_pb2` statements, so the
# `generated/` directory itself (not its parent) must be on sys.path.
_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
if _GEN not in sys.path:
    sys.path.insert(0, _GEN)

import grpc

logger = logging.getLogger(__name__)


class SoilQueryResult:
    __slots__ = (
        "trail_id",
        "outcome",
        "approach",
        "similarity",
        "success_rate",
        "agent_id",
        "task_domain",
        "task_summary",
        "hits",
        "strength",
        "resources",
    )

    def __init__(
        self,
        trail_id: str,
        outcome: str,
        approach: dict,
        similarity: float,
        success_rate: float,
        agent_id: str,
        task_domain: str,
        task_summary: str,
        hits: int,
        strength: float,
        resources: dict,
    ):
        self.trail_id = trail_id
        self.outcome = outcome
        self.approach = approach
        self.similarity = similarity
        self.success_rate = success_rate
        self.agent_id = agent_id
        self.task_domain = task_domain
        self.task_summary = task_summary
        self.hits = hits
        self.strength = strength
        self.resources = resources


# Connection-tuning options shared across SoilClient instances.
_GRPC_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 30_000),
    ("grpc.keepalive_timeout_ms", 10_000),
    ("grpc.keepalive_permit_without_calls", 1),
    # ~16 MB so big embeddings + future batched queries don't overflow.
    ("grpc.max_send_message_length", 16 * 1024 * 1024),
    ("grpc.max_receive_message_length", 16 * 1024 * 1024),
]


class SoilClient:
    def __init__(self):
        host = os.getenv("SOIL_GRPC_HOST", "localhost")
        port = os.getenv("SOIL_GRPC_PORT", "50051")
        self._address = f"{host}:{port}"
        self._channel: Optional[grpc.Channel] = None
        self._stub: Any = None
        self._soil_pb2: Any = None
        self._lock = threading.Lock()

    def connect(self):
        if self._stub is not None:
            return
        with self._lock:
            if self._stub is not None:
                return
            try:
                import soil_pb2  # type: ignore[import-not-found]
                import soil_pb2_grpc  # type: ignore[import-not-found]
            except ImportError as e:
                logger.error(
                    "Proto stubs not found. Run 'make proto' to generate them. Error: %s",
                    e,
                )
                raise
            self._soil_pb2 = soil_pb2
            self._channel = grpc.insecure_channel(self._address, options=_GRPC_CHANNEL_OPTIONS)
            self._stub = soil_pb2_grpc.SoilStub(self._channel)
            logger.info("SoilClient connected to %s", self._address)

    def query_similar(
        self,
        embedding: list[float],
        limit: int = 5,
        threshold: float = 0.7,
        domain: str = "",
    ) -> list[SoilQueryResult]:
        self.connect()
        try:
            request = self._soil_pb2.QueryRequest(
                embedding=embedding,
                limit=limit,
                threshold=threshold,
                domain=domain,
            )
            response = self._stub.QuerySimilar(request, timeout=10.0)
        except grpc.RpcError as e:
            logger.warning("SoilClient.query_similar failed: %s", e.details())
            return []

        results = []
        for r in response.results:
            try:
                approach = json.loads(r.approach) if r.approach else {}
                if not isinstance(approach, dict):
                    approach = {"raw": r.approach}
            except json.JSONDecodeError:
                approach = {"raw": r.approach}
            results.append(SoilQueryResult(
                trail_id=r.trail_id,
                outcome=r.outcome,
                approach=approach,
                similarity=getattr(r, "similarity", r.success_rate),
                success_rate=r.success_rate,
                agent_id=r.agent_id,
                task_domain=r.task_domain,
                task_summary=getattr(r, "task_summary", ""),
                hits=r.hits,
                strength=getattr(r, "strength", 1.0),
                resources=dict(r.resources),
            ))
        return results

    def leave_trail(
        self,
        embedding: list[float],
        outcome: str,
        approach: dict,
        agent_id: str,
        task_domain: str,
        task_summary: str,
        resources: Optional[dict] = None,
    ) -> bool:
        self.connect()
        resources = resources or {}
        try:
            request = self._soil_pb2.Trail(
                embedding=embedding,
                outcome=outcome,
                approach=json.dumps(approach),
                agent_id=agent_id,
                task_domain=task_domain,
                task_summary=task_summary,
                timestamp=self._now_iso(),
                resources={k: float(v) for k, v in resources.items()},
            )
            self._stub.LeaveTrail(request, timeout=10.0)
            logger.debug("Trail left: agent=%s domain=%s outcome=%s",
                         agent_id, task_domain, outcome)
            return True
        except grpc.RpcError as e:
            logger.warning("SoilClient.leave_trail failed: %s", e.details())
            return False

    def mark_failure(
        self,
        embedding: list[float],
        failed_approach: dict,
        error: str,
        agent_id: str,
        severity: str = "medium",
        task_domain: str = "",
        avoid_in_future: bool = True,
    ) -> bool:
        self.connect()
        try:
            request = self._soil_pb2.FailureMarker(
                embedding=embedding,
                failed_approach=json.dumps(failed_approach),
                error=error,
                agent_id=agent_id,
                severity=severity,
                task_domain=task_domain,
                avoid_in_future=avoid_in_future,
            )
            self._stub.MarkFailure(request, timeout=10.0)
            return True
        except grpc.RpcError as e:
            logger.warning("SoilClient.mark_failure failed: %s", e.details())
            return False

    def get_reputation(self, agent_id: str) -> dict:
        self.connect()
        try:
            request = self._soil_pb2.AgentRequest(agent_id=agent_id)
            rep = self._stub.GetReputation(request, timeout=5.0)
            return {
                "agent_id": rep.agent_id,
                "success_rate": rep.success_rate,
                "tasks_completed": rep.tasks_completed,
                "specialization": rep.specialization,
            }
        except grpc.RpcError as e:
            logger.warning("SoilClient.get_reputation failed: %s", e.details())
            return {}

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def close(self):
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
