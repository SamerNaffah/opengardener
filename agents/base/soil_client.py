"""
gRPC client for the Rust Soil service.
Wraps generated protobuf stubs with a clean Python interface.
"""

import os
import sys
import logging
from typing import Optional

# Add generated stubs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))

import grpc

logger = logging.getLogger(__name__)


class SoilQueryResult:
    def __init__(self, trail_id, outcome, approach, success_rate, agent_id, task_domain, hits, resources):
        self.trail_id = trail_id
        self.outcome = outcome
        self.approach = approach        # dict
        self.success_rate = success_rate
        self.agent_id = agent_id
        self.task_domain = task_domain
        self.hits = hits
        self.resources = resources


class SoilClient:
    def __init__(self):
        host = os.getenv("SOIL_GRPC_HOST", "localhost")
        port = os.getenv("SOIL_GRPC_PORT", "50051")
        self._address = f"{host}:{port}"
        self._channel = None
        self._stub = None

    def connect(self):
        if self._stub is not None:
            return

        try:
            from generated import soil_pb2, soil_pb2_grpc
            self._soil_pb2 = soil_pb2
            self._channel = grpc.insecure_channel(self._address)
            self._stub = soil_pb2_grpc.SoilStub(self._channel)
            logger.info(f"SoilClient connected to {self._address}")
        except ImportError as e:
            logger.error(
                f"Proto stubs not found. Run 'make proto' to generate them. Error: {e}"
            )
            raise

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

            results = []
            for r in response.results:
                import json
                try:
                    approach = json.loads(r.approach) if r.approach else {}
                except json.JSONDecodeError:
                    approach = {"raw": r.approach}

                results.append(SoilQueryResult(
                    trail_id=r.trail_id,
                    outcome=r.outcome,
                    approach=approach,
                    success_rate=r.success_rate,
                    agent_id=r.agent_id,
                    task_domain=r.task_domain,
                    hits=r.hits,
                    resources=dict(r.resources),
                ))
            return results

        except grpc.RpcError as e:
            logger.warning(f"SoilClient.query_similar failed: {e.details()}")
            return []

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

        try:
            import json
            resources = resources or {}
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
            logger.debug(f"Trail left: agent={agent_id} domain={task_domain} outcome={outcome}")
            return True

        except grpc.RpcError as e:
            logger.warning(f"SoilClient.leave_trail failed: {e.details()}")
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
            import json
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
            logger.warning(f"SoilClient.mark_failure failed: {e.details()}")
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
            logger.warning(f"SoilClient.get_reputation failed: {e.details()}")
            return {}

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def close(self):
        if self._channel:
            self._channel.close()
