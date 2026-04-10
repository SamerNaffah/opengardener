"""
Seed Tasks Script — bootstraps the Soil with initial task runs.

Runs one round of tasks across all three domains to:
1. Produce initial pheromone trails
2. Demonstrate stigmergic communication is working
3. Give agents something to learn from before the main loop

Usage:
  make seed          # via Docker Compose
  python scripts/seed_tasks.py  # locally (requires local soil connection)
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def seed_data_cleaning():
    from specialists.data_cleaner import DataCleanerAgent

    agent = DataCleanerAgent(agent_id="seed-data-001")

    data_dir = os.getenv("DATA_DIR", "/data")
    csv_path = os.path.join(data_dir, "sample", "dirty_data.csv")

    if not os.path.exists(csv_path):
        logger.warning(f"Sample CSV not found at {csv_path}. Skipping data cleaning seed.")
        return

    logger.info("=== Seeding DATA CLEANING tasks ===")

    # Run 5 cleaning tasks with different approaches to build diverse trails
    tasks = [
        {"schema": None},
        {"schema": {"id": "numeric", "age": "numeric", "score": "numeric"}},
        {"schema": None},
        {"schema": None},
        {"schema": {"id": "numeric", "score": "numeric"}},
    ]

    for i, task_args in enumerate(tasks):
        logger.info(f"Data cleaning task {i+1}/{len(tasks)}")
        try:
            result = agent.clean(csv_path, schema=task_args["schema"])
            logger.info(f"  Result: {result.get('method', 'unknown')} — rows_removed={result.get('rows_removed', 'N/A')}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
        time.sleep(1)


def seed_code_generation():
    from specialists.code_generator import CodeGeneratorAgent

    agent = CodeGeneratorAgent(agent_id="seed-code-001")

    logger.info("=== Seeding CODE GENERATION tasks ===")

    tasks = [
        "Write a Python function that validates an email address",
        "Write a Python function that parses JSON from a string safely",
        "Write a Python function that retries a function call up to N times with exponential backoff",
        "Write a Python class that implements a simple LRU cache",
        "Write a Python function that flattens a nested list",
    ]

    for i, task in enumerate(tasks):
        logger.info(f"Code generation task {i+1}/{len(tasks)}: {task[:60]}...")
        try:
            result = agent.generate(task)
            code_len = len(result.get("code", ""))
            logger.info(f"  Generated {code_len} chars via {result.get('method', 'unknown')}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
        time.sleep(2)  # LLM calls take longer


def seed_api_testing():
    from specialists.api_tester import ApiTesterAgent

    agent = ApiTesterAgent(agent_id="seed-api-001")

    endpoints_file = os.getenv("ENDPOINTS_FILE", "/data/sample/api_endpoints.json")

    if os.path.exists(endpoints_file):
        with open(endpoints_file) as f:
            endpoints = json.load(f)
    else:
        endpoints = [
            {"url": "https://httpbin.org/get", "method": "GET", "expected_status": 200},
            {"url": "https://jsonplaceholder.typicode.com/posts/1", "method": "GET", "expected_status": 200},
        ]

    logger.info("=== Seeding API TESTING tasks ===")

    for i, ep in enumerate(endpoints[:5]):  # first 5 only
        logger.info(f"API test {i+1}: {ep['method']} {ep['url']}")
        try:
            result = agent.test_endpoint(
                url=ep["url"],
                method=ep.get("method", "GET"),
                expected_status=ep.get("expected_status", 200),
            )
            logger.info(f"  Status={result.get('status_code')} success={result.get('success')}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
        time.sleep(1)


def main():
    logger.info("OpenGardener Seed Runner starting...")

    startup_delay = float(os.getenv("STARTUP_DELAY_S", "5"))
    if startup_delay > 0:
        logger.info(f"Waiting {startup_delay}s for services to be ready...")
        time.sleep(startup_delay)

    seed_data_cleaning()
    seed_code_generation()
    seed_api_testing()

    logger.info("Seeding complete. Soil has been initialized with pheromone trails.")
    logger.info("Run 'make observe' to inspect the trails.")


if __name__ == "__main__":
    main()
