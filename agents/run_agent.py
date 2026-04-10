"""
Agent entry point for Docker containers.

Environment variables:
  AGENT_TYPE    - "data_cleaner" | "code_generator" | "api_tester"
  AGENT_ID      - optional explicit agent ID (auto-generated if not set)
  TASK_LOOP     - "true" (default) to run continuous loop, "false" for single task
"""

import json
import logging
import os
import sys
import time
import random

# Configure logging before importing anything else
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add agents dir to path
sys.path.insert(0, os.path.dirname(__file__))


def load_agent():
    agent_type = os.getenv("AGENT_TYPE", "data_cleaner").lower()
    agent_id = os.getenv("AGENT_ID")

    if agent_type == "data_cleaner":
        from specialists.data_cleaner import DataCleanerAgent
        return DataCleanerAgent(agent_id=agent_id)

    elif agent_type == "code_generator":
        from specialists.code_generator import CodeGeneratorAgent
        return CodeGeneratorAgent(agent_id=agent_id)

    elif agent_type == "api_tester":
        from specialists.api_tester import ApiTesterAgent
        return ApiTesterAgent(agent_id=agent_id)

    else:
        raise ValueError(f"Unknown AGENT_TYPE: {agent_type}. Use: data_cleaner, code_generator, api_tester")


def run_data_cleaner_loop(agent):
    """Continuous loop for data cleaning tasks."""
    from pathlib import Path
    import glob

    data_dir = os.getenv("DATA_DIR", "/data")
    csv_files = glob.glob(f"{data_dir}/**/*.csv", recursive=True)

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}. Waiting for tasks...")
        time.sleep(30)
        return

    # Pick a random CSV to clean
    file_path = random.choice(csv_files)
    logger.info(f"Cleaning: {file_path}")
    result = agent.clean(file_path)
    logger.info(f"Result: {json.dumps(result, indent=2)}")


def run_code_generator_loop(agent):
    """Continuous loop for code generation tasks."""
    tasks_file = os.getenv("TASKS_FILE", "/tasks/code_tasks.json")

    if not os.path.exists(tasks_file):
        # Generate a random task if no task file
        tasks = [
            "Write a Python function that calculates the Fibonacci sequence up to n",
            "Write a Python class that implements a thread-safe queue",
            "Write a Python function that parses a CSV file and returns a list of dicts",
            "Write a Python decorator that measures and logs function execution time",
            "Write a Python function that validates an email address using regex",
        ]
        task = random.choice(tasks)
    else:
        with open(tasks_file) as f:
            task_list = json.load(f)
        task = random.choice(task_list)["description"]

    logger.info(f"Generating code for: {task[:80]}...")
    result = agent.generate(task)
    if result.get("code"):
        logger.info(f"Generated {len(result['code'])} chars of {result.get('language', 'code')}")
    else:
        logger.warning(f"Code generation returned no code: {result}")


def run_api_tester_loop(agent):
    """Continuous loop for API testing tasks."""
    endpoints_file = os.getenv("ENDPOINTS_FILE", "/data/api_endpoints.json")

    if os.path.exists(endpoints_file):
        with open(endpoints_file) as f:
            endpoints = json.load(f)
    else:
        # Default public endpoints for testing
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


def main():
    agent_type = os.getenv("AGENT_TYPE", "data_cleaner").lower()
    task_loop = os.getenv("TASK_LOOP", "true").lower() == "true"
    loop_delay = float(os.getenv("LOOP_DELAY_S", "5"))

    logger.info(f"Starting agent: type={agent_type}, loop={task_loop}")

    # Brief startup delay to allow Gardener/Soil to be ready
    startup_delay = float(os.getenv("STARTUP_DELAY_S", "10"))
    logger.info(f"Waiting {startup_delay}s for services to be ready...")
    time.sleep(startup_delay)

    agent = load_agent()

    loop_fn = {
        "data_cleaner": run_data_cleaner_loop,
        "code_generator": run_code_generator_loop,
        "api_tester": run_api_tester_loop,
    }.get(agent_type, run_data_cleaner_loop)

    if task_loop:
        logger.info(f"Agent {agent.id} entering task loop (delay={loop_delay}s between tasks)")
        while True:
            try:
                loop_fn(agent)
            except KeyboardInterrupt:
                logger.info("Agent shutting down.")
                break
            except Exception as e:
                logger.error(f"Unhandled error in task loop: {e}", exc_info=True)

            time.sleep(loop_delay + random.uniform(0, 2))  # jitter
    else:
        # Single task mode (used by seed-runner)
        loop_fn(agent)
        logger.info(f"Agent {agent.id} completed single task. Exiting.")


if __name__ == "__main__":
    main()
