"""
ApiTesterAgent — specializes in HTTP API endpoint testing.

Strategies (emergent via soil trails):
  - basic_get: simple GET request, check status code
  - auth_bearer: GET with Bearer token
  - post_json: POST with JSON payload
  - health_check: lightweight ping (HEAD or GET /)
  - retry_backoff: retry with exponential backoff on failure

Round-2 eval additions:
  - test_endpoint() accepts difficulty: float
  - FAILURE_INJECTION_RATE env var: with probability = difficulty × rate,
    hit a known-bad URL so the agent sees real failures on hard tasks
"""

import json
import logging
import os
import random
import time
from typing import Optional

import httpx

from base.agent import EmergentAgent, TaskOutcome

logger = logging.getLogger(__name__)

_FAILURE_URLS = [
    "https://httpbin.org/status/500",
    "https://httpbin.org/status/503",
]


class ApiTesterAgent(EmergentAgent):
    def __init__(self, agent_id: Optional[str] = None):
        genome = {
            "specialization": "api_testing",
            "default_approach": {
                "method": "basic_get",
                "timeout_s": 10,
                "expected_status": 200,
            },
            "fallback_approach": {
                "method": "health_check",
                "timeout_s": 5,
            },
        }
        super().__init__(agent_id=agent_id, genome=genome)

    @property
    def task_domain(self) -> str:
        return "api_testing"

    def test_endpoint(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[dict] = None,
        payload: Optional[dict] = None,
        expected_status: int = 200,
        difficulty: float = 0.0,
    ) -> dict:
        """
        Test an HTTP endpoint using the strategy from the Soil.
        difficulty: 0-1; combined with FAILURE_INJECTION_RATE env var to inject failures.
        """
        # Failure injection: redirect to a bad URL proportional to difficulty
        rate = float(os.getenv("FAILURE_INJECTION_RATE", "0.0"))
        if rate > 0 and difficulty > 0 and random.random() < difficulty * rate:
            url = random.choice(_FAILURE_URLS)
            expected_status = 200  # will fail against 500/503
            logger.debug("[%s] Failure injection: redirected to %s", self.id, url)

        task_desc = f"test {method} {url} (expected {expected_status})"
        strategy = self.before_task(task_desc)
        start = time.time()

        try:
            result = self._execute_strategy(url, strategy, headers, payload, expected_status)
            elapsed_ms = (time.time() - start) * 1000
            result["elapsed_ms"] = elapsed_ms

            success = result.get("status_code") == result.get("expected_status", expected_status)

            self.after_task(
                task_desc,
                outcome=TaskOutcome(success=success, data=result),
                resources={"cpu_ms": elapsed_ms},
            )
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning(f"[{self.id}] API test failed: {e}")

            self.after_task(
                task_desc,
                outcome=TaskOutcome(success=False, error=str(e)),
                resources={"cpu_ms": elapsed_ms},
            )
            return {"success": False, "error": str(e), "url": url}

    def _execute_strategy(
        self,
        url: str,
        strategy: dict,
        headers: Optional[dict],
        payload: Optional[dict],
        expected_status: int,
    ) -> dict:
        method_name = strategy.get("method", "basic_get")
        timeout = strategy.get("timeout_s", 10)
        headers = headers or {}

        if method_name == "basic_get":
            return self._basic_get(url, headers, timeout, expected_status)
        elif method_name == "auth_bearer":
            token = strategy.get("token") or os.getenv("API_TEST_TOKEN", "test-token")
            headers["Authorization"] = f"Bearer {token}"
            return self._basic_get(url, headers, timeout, expected_status)
        elif method_name == "post_json":
            return self._post_json(url, headers, payload or {}, timeout, expected_status)
        elif method_name == "health_check":
            return self._health_check(url, timeout)
        elif method_name == "retry_backoff":
            return self._retry_with_backoff(url, headers, timeout, expected_status, strategy)
        else:
            return self._basic_get(url, headers, timeout, expected_status)

    def _basic_get(self, url: str, headers: dict, timeout: float, expected_status: int) -> dict:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)

        return {
            "url": url,
            "method": "GET",
            "status_code": response.status_code,
            "expected_status": expected_status,
            "success": response.status_code == expected_status,
            "response_size_bytes": len(response.content),
            "headers": dict(response.headers),
        }

    def _post_json(self, url: str, headers: dict, payload: dict, timeout: float, expected_status: int) -> dict:
        headers.setdefault("Content-Type", "application/json")
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(url, json=payload, headers=headers)

        return {
            "url": url,
            "method": "POST",
            "status_code": response.status_code,
            "expected_status": expected_status,
            "success": response.status_code == expected_status,
            "response_size_bytes": len(response.content),
        }

    def _health_check(self, url: str, timeout: float) -> dict:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            try:
                response = client.head(url)
            except Exception:
                response = client.get(url)

        reachable = response.status_code < 500
        return {
            "url": url,
            "method": "HEAD",
            "status_code": response.status_code,
            "expected_status": 200,
            "success": reachable,
            "reachable": reachable,
        }

    def _retry_with_backoff(
        self, url: str, headers: dict, timeout: float, expected_status: int, strategy: dict
    ) -> dict:
        max_retries = int(strategy.get("max_retries", 3))
        backoff_s = float(strategy.get("backoff_s", 1.0))
        last_result = {}

        for attempt in range(max_retries):
            try:
                result = self._basic_get(url, headers, timeout, expected_status)
                if result["success"]:
                    result["attempts"] = attempt + 1
                    return result
                last_result = result
            except Exception as e:
                last_result = {"error": str(e), "success": False}

            if attempt < max_retries - 1:
                time.sleep(backoff_s * (2 ** attempt))

        last_result["attempts"] = max_retries
        last_result["method"] = "retry_backoff"
        return last_result
