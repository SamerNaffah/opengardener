"""
GenericAgent — cross-domain dispatcher.

Infers domain from soil similarity queries across all three domains.
Used in Round-2 eval to test whether agents can specialise emergently
without being pinned to a domain.

AC2' and AC2'b measure this agent's specialisation_index at task 100
(should be ≤ 0.5 — not yet specialised) and task 500 (should be ≥ 0.6).
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Optional

from base.agent import EmergentAgent, TaskOutcome

logger = logging.getLogger(__name__)

_ALL_DOMAINS = ["data_cleaning", "code_generation", "api_testing"]


class GenericAgent(EmergentAgent):
    def __init__(self, agent_id: Optional[str] = None):
        genome = {
            "specialization": "generic",
            "default_approach": {"method": "dispatch", "domain": None},
        }
        super().__init__(agent_id=agent_id, genome=genome)
        self._inferred_domain: Optional[str] = None
        self._domain_history: Counter = Counter()

    @property
    def task_domain(self) -> str:
        # Returns inferred domain for soil deposits; "generic" before first inference
        return self._inferred_domain or "generic"

    def execute(self, task: dict) -> dict:
        """
        Execute a task by:
        1. Querying soil across all domains to find the best-matching domain
        2. Dispatching to the appropriate specialist method
        3. Tracking domain history (used for external specialisation measurement)
        """
        description = task.get("description", "")
        ground_truth_domain = task.get("domain")
        difficulty = float(task.get("difficulty", 0.0))
        verifier = task.get("verifier") or ""

        # Infer domain from soil; fall back to ground truth when no trails exist yet
        inferred = self._infer_domain(description)
        self._inferred_domain = inferred or ground_truth_domain or "data_cleaning"
        self._domain_history[self._inferred_domain] += 1

        logger.debug(
            "[%s] domain inferred=%s ground_truth=%s",
            self.id, self._inferred_domain, ground_truth_domain,
        )

        try:
            result = self._dispatch(description, difficulty, verifier)
        except Exception as e:
            logger.warning("[%s] dispatch failed: %s", self.id, e)
            result = {"success": False, "error": str(e)}

        return result

    def _infer_domain(self, description: str) -> Optional[str]:
        """
        Query soil across all three domains. Pick domain with highest mean
        similarity among successful trails. Returns None when no trails exist.
        """
        embedding = self.embed(description)
        thresh = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

        best_domain: Optional[str] = None
        best_score = -1.0

        for domain in _ALL_DOMAINS:
            trails = self.soil.query_similar(
                embedding=embedding,
                limit=3,
                threshold=thresh,
                domain=domain,
            )
            successes = [t for t in trails if t.outcome == "success"]
            if successes:
                avg = sum(t.similarity for t in successes) / len(successes)
                if avg > best_score:
                    best_score = avg
                    best_domain = domain

        return best_domain

    def _dispatch(self, description: str, difficulty: float, verifier: str) -> dict:
        """Dispatch to the appropriate specialist logic based on inferred domain."""
        domain = self._inferred_domain or "api_testing"

        if domain == "data_cleaning":
            from specialists.data_cleaner import DataCleanerAgent
            csv_path = os.path.join(
                os.getenv("DATA_DIR", "/data"), "sample", "dirty_data.csv"
            )
            # Reuse soil connection — pass self's soil/llm into a lightweight wrapper
            agent = self._make_sub(DataCleanerAgent)
            return agent.clean(csv_path, difficulty=difficulty)

        elif domain == "code_generation":
            from specialists.code_generator import CodeGeneratorAgent
            agent = self._make_sub(CodeGeneratorAgent)
            return agent.generate(description, verifier=verifier)

        else:  # api_testing
            from specialists.api_tester import ApiTesterAgent
            agent = self._make_sub(ApiTesterAgent)
            return agent.test_endpoint(
                "https://httpbin.org/get",
                difficulty=difficulty,
            )

    def _make_sub(self, cls):
        """
        Create a specialist instance that shares this agent's soil/llm connections.
        Uses __new__ to skip __init__ (which would create new gRPC channels), then
        copies the relevant runtime state.
        """
        sub = cls.__new__(cls)
        # Copy all instance attributes — soil, llm, genome, experience_buffer, etc.
        sub.__dict__.update(self.__dict__)
        # Override id so soil trails are attributed to the generic agent
        sub._id = self._id
        return sub
