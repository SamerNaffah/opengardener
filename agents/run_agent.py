"""
Agent entry point for Docker containers.

Environment variables:
  AGENT_TYPE     - "data_cleaner" | "code_generator" | "api_tester"
  AGENT_ID       - optional explicit agent ID (auto-generated if not set)
  TASK_LOOP      - "true" (default) to run continuous loop, "false" for single task
  STARTUP_DELAY_S, LOOP_DELAY_S — float seconds (defaults 10, 5)
  DATA_DIR, ENDPOINTS_FILE, TASKS_FILE — domain-specific input paths

Stops cleanly on SIGTERM/SIGINT (Docker stop + Ctrl-C).
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from typing import Callable, Optional

# Configure logging before importing anything else
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add agents dir to path
sys.path.insert(0, os.path.dirname(__file__))


# ─── Cooperative shutdown event (set by SIGTERM/SIGINT) ────────────────────
_stop = threading.Event()


def _request_stop(signum, _frame):
    name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    logger.info(f"Received {name}, requesting graceful shutdown.")
    _stop.set()


def _interruptible_sleep(seconds: float) -> bool:
    """Sleep up to `seconds`, returning early (True) if shutdown was requested."""
    return _stop.wait(timeout=seconds)


# ─── Agent factory ─────────────────────────────────────────────────────────
def load_agent():
    agent_type = os.getenv("AGENT_TYPE", "data_cleaner").lower()
    agent_id = os.getenv("AGENT_ID")

    if agent_type == "data_cleaner":
        from specialists.data_cleaner import DataCleanerAgent
        return DataCleanerAgent(agent_id=agent_id)
    if agent_type == "code_generator":
        from specialists.code_generator import CodeGeneratorAgent
        return CodeGeneratorAgent(agent_id=agent_id)
    if agent_type == "api_tester":
        from specialists.api_tester import ApiTesterAgent
        return ApiTesterAgent(agent_id=agent_id)
    raise ValueError(
        f"Unknown AGENT_TYPE: {agent_type}. Use: data_cleaner, code_generator, api_tester"
    )


# ─── Per-domain task loops ─────────────────────────────────────────────────
def run_data_cleaner_loop(agent):
    data_dir = os.getenv("DATA_DIR", "/data")
    csv_files = glob.glob(f"{data_dir}/**/*.csv", recursive=True)
    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}. Waiting for tasks...")
        _interruptible_sleep(30)
        return
    file_path = random.choice(csv_files)
    logger.info(f"Cleaning: {file_path}")
    result = agent.clean(file_path)
    logger.info(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")


def run_code_generator_loop(agent):
    tasks_file = os.getenv("TASKS_FILE", "/tasks/code_tasks.json")
    if os.path.exists(tasks_file):
        with open(tasks_file) as f:
            task_list = json.load(f)
        task = random.choice(task_list)["description"]
    else:
        tasks = [
            "Write a Python function that calculates the Fibonacci sequence up to n",
            "Write a Python class that implements a thread-safe queue",
            "Write a Python function that parses a CSV file and returns a list of dicts",
            "Write a Python decorator that measures and logs function execution time",
            "Write a Python function that validates an email address using regex",
        ]
        task = random.choice(tasks)
    logger.info(f"Generating code for: {task[:80]}...")
    result = agent.generate(task)
    if result.get("code"):
        logger.info(f"Generated {len(result['code'])} chars of {result.get('language', 'code')}")
    else:
        logger.warning(f"Code generation returned no code: {result}")


def run_api_tester_loop(agent):
    endpoints_file = os.getenv("ENDPOINTS_FILE", "/data/api_endpoints.json")
    if os.path.exists(endpoints_file):
        with open(endpoints_file) as f:
            endpoints = json.load(f)
    else:
        endpoints = [
            {"url": "https://httpbin.org/get", "method": "GET", "expected_status": 200},
            {"url": "https://httpbin.org/status/200", "method": "GET", "expected_status": 200},
            {"url": "https://httpbin.org/status/404", "method": "GET", "expected_status": 404},
            {"url": "https://jsonplaceholder.typicode.com/posts/1", "method": "GET", "expected_status": 200},
        ]
    endpoint = random.choice(endpoints)
    logger.info(f"Testing: {endpoint['method']} {endpoint['url']}")
    result = agent.test_endpoint(
        url=endpoint["url"],
        method=endpoint.get("method", "GET"),
        expected_status=endpoint.get("expected_status", 200),
    )
    logger.info(f"Result: status={result.get('status_code')} success={result.get('success')}")


LOOPS: dict[str, Callable] = {
    "data_cleaner": run_data_cleaner_loop,
    "code_generator": run_code_generator_loop,
    "api_tester": run_api_tester_loop,
}


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    agent_type = os.getenv("AGENT_TYPE", "data_cleaner").lower()
    task_loop = os.getenv("TASK_LOOP", "true").lower() == "true"
    loop_delay = float(os.getenv("LOOP_DELAY_S", "5"))

    logger.info(f"Starting agent: type={agent_type}, loop={task_loop}")

    startup_delay = float(os.getenv("STARTUP_DELAY_S", "10"))
    if startup_delay > 0:
        logger.info(f"Waiting {startup_delay}s for services to be ready...")
        if _interruptible_sleep(startup_delay):
            logger.info("Shutdown requested during startup; exiting.")
            return

    agent = load_agent()
    loop_fn = LOOPS.get(agent_type, run_data_cleaner_loop)

    if not task_loop:
        loop_fn(agent)
        logger.info(f"Agent {agent.id} completed single task. Exiting.")
        return

    logger.info(f"Agent {agent.id} entering task loop (delay={loop_delay}s ± jitter)")
    while not _stop.is_set():
        try:
            loop_fn(agent)
            # Agent may have set self.terminated=True via prune signal —
            # respect that immediately rather than running another task.
            if getattr(agent, "terminated", False):
                logger.info(f"Agent {agent.id} terminated by Gardener; exiting loop.")
                break
        except SystemExit:
            # Raised by EmergentAgent._archive_and_exit — exit cleanly.
            logger.info(f"Agent {agent.id} exiting per archive_and_exit.")
            break
        except KeyboardInterrupt:
            logger.info("Agent shutting down on KeyboardInterrupt.")
            break
        except Exception:
            logger.exception("Unhandled error in task loop")

        _interruptible_sleep(loop_delay + random.uniform(0, 2))

    # Best-effort cleanup of the soil channel.
    try:
        agent.soil.close()
    except Exception:
        pass
    logger.info("Agent loop exited.")


if __name__ == "__main__":
    main()
