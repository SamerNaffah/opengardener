"""
Synthetic task corpus generator for OpenGardener evaluation.

Produces a JSONL file where each line is one task:
  {"id": "t0001", "domain": "data_cleaning", "description": "...",
   "difficulty": 0.3, "expected_method": "pandas_dropna", "seed_idx": 42}

Design choices:
  - Three domains, balanced by default (configurable via --domain-weights).
  - Difficulty is a float in [0, 1]; higher → task description is more
    ambiguous / the "correct" method is less obvious from the text alone.
  - A fixed OG_SEED (env or --seed) makes runs reproducible.
  - Default corpus size is 1 200 tasks (enough to show specialisation curves).

Usage:
  python scripts/generate_tasks.py                          # 1200 tasks, seed from OG_SEED env
  python scripts/generate_tasks.py --n 500 --seed 42       # quick smoke test
  python scripts/generate_tasks.py --out data/generated/tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# ─── Domain templates ───────────────────────────────────────────────────────

_DATA_CLEANING_TEMPLATES = [
    # (description template, expected_method, base_difficulty)
    ("clean CSV file {filename} by dropping null rows and deduplicating", "pandas_dropna", 0.1),
    ("remove null values from {filename}", "pandas_dropna", 0.15),
    ("deduplicate rows in {filename} and strip whitespace", "pandas_dropna", 0.2),
    ("apply regex patterns to sanitize {filename}: remove control chars and excess spaces", "regex_cleaning", 0.25),
    ("strip non-printable characters from {filename} using regex substitution", "regex_cleaning", 0.3),
    ("fill missing numeric values in {filename} using {fill_strategy}", "fill_missing", 0.35),
    ("impute NaN entries in {filename} with {fill_strategy} strategy", "fill_missing", 0.4),
    ("validate {filename} against schema: columns {cols}", "schema_validation", 0.45),
    ("check {filename} conforms to schema with required numeric cols {cols}", "schema_validation", 0.5),
    # harder — more ambiguous phrasing
    ("process {filename} to improve data quality", "pandas_dropna", 0.6),
    ("fix {filename} so downstream ML pipeline does not fail", "fill_missing", 0.65),
    ("prepare {filename} for statistical analysis", "pandas_dropna", 0.7),
    ("normalise and clean {filename} for reporting", "regex_cleaning", 0.75),
    ("handle outliers and missing data in {filename}", "fill_missing", 0.8),
    ("make {filename} consistent with data dictionary {cols}", "schema_validation", 0.85),
]

_CODE_GENERATION_TEMPLATES = [
    ("write a Python function that validates an email address", "llm_direct", 0.1),
    ("write a Python function that parses JSON from a string safely", "llm_direct", 0.15),
    ("write a Python function that retries a call up to {n} times with exponential backoff", "llm_direct", 0.2),
    ("write a Python class that implements an LRU cache of size {n}", "llm_direct", 0.25),
    ("write a Python function that flattens a nested list to depth {n}", "llm_direct", 0.3),
    ("write a Python decorator that memoizes function results", "llm_direct", 0.3),
    ("generate a Python {lang} function that computes {algo}", "template_based", 0.35),
    ("write a {lang} utility to {action} strings", "template_based", 0.4),
    ("implement a thread-safe singleton in Python", "llm_direct", 0.45),
    ("write a Python context manager for {resource} lifecycle", "llm_direct", 0.5),
    ("refactor this function to improve readability: handle {action} with {n} retries", "iterative_refinement", 0.55),
    ("write unit tests for a {lang} function that {action}", "iterative_refinement", 0.6),
    ("implement {algo} algorithm in Python with O({complexity}) complexity", "llm_direct", 0.65),
    ("generate boilerplate for a Python {lang} module that {action}", "template_based", 0.7),
    ("write idiomatic Python to {action} — optimise for readability", "llm_with_context", 0.75),
    ("port this logic from {lang} to Python: {action}", "llm_with_context", 0.8),
]

_API_TESTING_TEMPLATES = [
    ("test GET {url} expect status {status}", "basic_get", 0.1),
    ("verify {url} returns {status} on GET", "basic_get", 0.15),
    ("check that {url} responds within {timeout}s", "basic_get", 0.2),
    ("test {url} with Bearer token auth", "auth_bearer", 0.25),
    ("authenticate to {url} and verify {status}", "auth_bearer", 0.3),
    ("POST JSON payload to {url} and expect {status}", "post_json", 0.3),
    ("send {method} request to {url} with body and check {status}", "post_json", 0.35),
    ("health check {url} with a lightweight ping", "health_check", 0.2),
    ("confirm {url} liveness probe returns {status}", "health_check", 0.25),
    ("test {url} with retry on transient failures, backoff {timeout}s", "retry_backoff", 0.45),
    ("verify {url} eventually returns {status} under load", "retry_backoff", 0.5),
    ("run a smoke test suite against {url}", "basic_get", 0.55),
    ("test authentication flow against {url} with token", "auth_bearer", 0.6),
    ("validate API contract for {method} {url}", "post_json", 0.65),
    ("stress test {url} with {n} sequential requests", "retry_backoff", 0.7),
    ("integration test: call {url} then {url2} and verify chained response", "post_json", 0.8),
]

# ─── Parameter pools ────────────────────────────────────────────────────────

_FILENAMES = [
    "customers.csv", "transactions.csv", "events.csv", "users.csv",
    "orders.csv", "products.csv", "logs.csv", "survey_responses.csv",
    "metrics.csv", "inventory.csv",
]
_FILL_STRATEGIES = ["median", "mean", "zero", "mode", "forward-fill"]
_SCHEMA_COLS = [
    "id, age, score", "user_id, amount", "timestamp, value, category",
    "name, email, created_at", "product_id, price, quantity",
]
_LANGS = ["Go", "Rust", "JavaScript", "TypeScript", "Java", "C#"]
_ALGOS = [
    "fibonacci", "quicksort", "binary search", "BFS graph traversal",
    "SHA-256 hash", "sliding window maximum", "topological sort",
]
_ACTIONS = [
    "sanitise", "parse", "transform", "validate", "format", "encode",
    "compress", "deduplicate", "aggregate", "filter",
]
_COMPLEXITIES = ["n log n", "n²", "n", "log n", "1"]
_RESOURCES = ["database connection", "file handle", "HTTP session", "thread pool", "lock"]
_URLS = [
    "https://api.example.com/v1/users",
    "https://api.example.com/v1/orders",
    "https://staging.internal/health",
    "https://jsonplaceholder.typicode.com/posts",
    "https://httpbin.org/get",
    "https://api.service.io/v2/data",
]
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
_STATUSES = [200, 201, 204, 400, 401, 403, 404, 500]
_NS = [3, 5, 10, 20, 50, 100]
_TIMEOUTS = [1, 2, 5, 10, 30]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        filename=rng.choice(_FILENAMES),
        fill_strategy=rng.choice(_FILL_STRATEGIES),
        cols=rng.choice(_SCHEMA_COLS),
        lang=rng.choice(_LANGS),
        algo=rng.choice(_ALGOS),
        action=rng.choice(_ACTIONS),
        complexity=rng.choice(_COMPLEXITIES),
        resource=rng.choice(_RESOURCES),
        url=rng.choice(_URLS),
        url2=rng.choice(_URLS),
        method=rng.choice(_METHODS),
        status=rng.choice(_STATUSES),
        timeout=rng.choice(_TIMEOUTS),
        n=rng.choice(_NS),
    )


# ─── Generator ──────────────────────────────────────────────────────────────

DOMAIN_TEMPLATES = {
    "data_cleaning": _DATA_CLEANING_TEMPLATES,
    "code_generation": _CODE_GENERATION_TEMPLATES,
    "api_testing": _API_TESTING_TEMPLATES,
}


def generate_corpus(
    n: int = 1200,
    seed: int = 0,
    domain_weights: dict[str, float] | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    weights = domain_weights or {"data_cleaning": 1.0, "code_generation": 1.0, "api_testing": 1.0}
    domains = list(weights.keys())
    w_vals = [weights[d] for d in domains]

    tasks = []
    for i in range(n):
        domain = rng.choices(domains, weights=w_vals, k=1)[0]
        templates = DOMAIN_TEMPLATES[domain]
        tmpl, expected_method, base_difficulty = rng.choice(templates)

        # Add jitter to difficulty (±0.05, clamped to [0,1])
        difficulty = max(0.0, min(1.0, base_difficulty + rng.uniform(-0.05, 0.05)))

        try:
            description = _fill(tmpl, rng)
        except KeyError:
            description = tmpl  # fallback: use template verbatim

        tasks.append({
            "id": f"t{i:05d}",
            "domain": domain,
            "description": description,
            "difficulty": round(difficulty, 3),
            "expected_method": expected_method,
            "seed_idx": i,
        })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OpenGardener task corpus")
    parser.add_argument("--n", type=int, default=1200, help="Number of tasks (default 1200)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (overrides OG_SEED env)")
    parser.add_argument(
        "--out",
        default="data/generated/tasks.jsonl",
        help="Output JSONL path (default: data/generated/tasks.jsonl)",
    )
    args = parser.parse_args()

    seed = args.seed
    if seed is None:
        env_seed = os.getenv("OG_SEED")
        seed = int(env_seed) if env_seed is not None else 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n} tasks with seed={seed} → {out_path}")
    tasks = generate_corpus(n=args.n, seed=seed)

    with out_path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    domain_counts = {}
    for t in tasks:
        domain_counts[t["domain"]] = domain_counts.get(t["domain"], 0) + 1

    print(f"Done. Distribution: {domain_counts}")


if __name__ == "__main__":
    main()
