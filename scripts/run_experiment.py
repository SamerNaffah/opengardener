"""
OpenGardener evaluation experiment runner.

Runs three conditions in sequence against real agents:
  CONTROL   — EXPLOIT_DISABLED=true  (always explore; no soil feedback)
  TREATMENT — EXPLOIT_DISABLED=false (stigmergic specialisation on)
  GENERIC   — EXPLOIT_DISABLED=false, GenericAgent (cross-domain; tests AC2'/AC2'b)

For each condition the script:
  1. Boots agents (or connects to already-running ones).
  2. Feeds tasks from the corpus one-by-one via the agent's task method.
  3. After each task, polls /metrics/eval and appends a record to a JSONL log.
  4. Stops after N tasks or when all tasks are consumed.

Outputs (written to --out-dir, default eval/runs/):
  control_<seed>.jsonl    — per-task log for control condition
  treatment_<seed>.jsonl  — per-task log for treatment condition

Each JSONL line:
  {
    "task_id": "t00042",
    "condition": "control" | "treatment",
    "domain": "data_cleaning",
    "task_seq": 42,           # global task counter
    "success": true,
    "agent_id": "agent-abc123",
    "strategy_method": "pandas_dropna",
    "elapsed_ms": 14.3,
    "soil_trails": 87,        # from /metrics/eval at this point
    "trails_per_domain": {"data_cleaning": 40, ...},
    "specialisation_index": 0.72,  # fraction of agent's tasks in its dominant domain
    "entropy": 1.21,               # Shannon entropy of (domain, method) distribution
    "timestamp": "2026-05-02T..."
  }

Usage:
  # Run both conditions, 500 tasks each, seeds 0 and 1:
  python scripts/run_experiment.py --tasks data/generated/tasks.jsonl --n 500 --seeds 0,1

  # Quick smoke (50 tasks, single seed):
  python scripts/run_experiment.py --n 50 --seeds 0 --smoke

Environment:
  OG_SEED              default seed if --seeds not given
  GARDENER_HTTP_HOST   default localhost
  GARDENER_HTTP_PORT   default 8080
  SOIL_GRPC_HOST / SOIL_GRPC_PORT
  GARDENER_GRPC_HOST / GARDENER_GRPC_PORT
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Add agents dir to path ─────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "agents"))


# ─── /metrics/eval HTTP client ──────────────────────────────────────────────

def _fetch_eval_metrics(host: str, port: int, timeout: float = 5.0) -> dict:
    url = f"http://{host}:{port}/metrics/eval"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug("metrics/eval unavailable (%s) — using zeros", e)
        return {}


# ─── Specialisation index ────────────────────────────────────────────────────

def _specialisation_index(agent_domain_counts: dict[str, dict[str, int]]) -> float:
    """
    Mean over all agents of (tasks_in_dominant_domain / total_tasks).
    An agent with 0 tasks contributes 0.
    """
    indices = []
    for counts in agent_domain_counts.values():
        total = sum(counts.values())
        if total == 0:
            continue
        dominant = max(counts.values())
        indices.append(dominant / total)
    return float(sum(indices) / len(indices)) if indices else 0.0


def _shannon_entropy(method_counts: Counter) -> float:
    total = sum(method_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in method_counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ─── Condition runner ────────────────────────────────────────────────────────

def run_condition(
    condition: str,
    tasks: list[dict],
    n: int,
    seed: int,
    out_path: Path,
    metrics_host: str,
    metrics_port: int,
    exploit_disabled: bool,
) -> list[dict]:
    os.environ["EXPLOIT_DISABLED"] = "true" if exploit_disabled else "false"
    os.environ["OG_SEED"] = str(seed)

    # Import agents after env vars are set so they pick up OG_SEED / EXPLOIT_DISABLED.
    import importlib
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("base.") or mod_name.startswith("specialists."):
            del sys.modules[mod_name]

    from specialists.data_cleaner import DataCleanerAgent
    from specialists.code_generator import CodeGeneratorAgent
    from specialists.api_tester import ApiTesterAgent

    agents = {
        "data_cleaning": DataCleanerAgent(agent_id=f"{condition}-dc-{seed}"),
        "code_generation": CodeGeneratorAgent(agent_id=f"{condition}-cg-{seed}"),
        "api_testing": ApiTesterAgent(agent_id=f"{condition}-at-{seed}"),
    }

    # Per-agent domain distribution (for specialisation index)
    agent_domain_counts: dict[str, dict[str, int]] = {
        aid: defaultdict(int) for aid in agents
    }
    # Global (domain, method) counts for entropy
    method_counts: Counter = Counter()

    records: list[dict] = []
    import random as _rand
    _rng = _rand.Random(seed ^ (0xC0FFEE if exploit_disabled else 0xBEEF))
    selected_tasks = tasks[:n]
    _rng.shuffle(selected_tasks)  # each condition sees tasks in different order

    logger.info("Condition=%s seed=%d tasks=%d exploit_disabled=%s", condition, seed, len(selected_tasks), exploit_disabled)

    with out_path.open("w") as fh:
        for seq, task in enumerate(selected_tasks):
            domain = task["domain"]
            agent_key = domain
            agent = agents.get(agent_key)
            if agent is None:
                continue

            t0 = time.time()
            success = False
            method = "unknown"

            difficulty = float(task.get("difficulty", 0.0))
            verifier = task.get("verifier") or ""

            try:
                if domain == "data_cleaning":
                    csv_path = str(_REPO / "data" / "sample" / "dirty_data.csv")
                    result = agent.clean(csv_path, difficulty=difficulty)
                    success = bool(result.get("success", False))
                    method = result.get("method", agent.current_strategy.get("method", "unknown"))

                elif domain == "code_generation":
                    result = agent.generate(task["description"], verifier=verifier)
                    success = bool(result.get("success", False))
                    method = result.get("method", agent.current_strategy.get("method", "unknown"))

                elif domain == "api_testing":
                    url = "https://httpbin.org/get"
                    result = agent.test_endpoint(url, method="GET", expected_status=200, difficulty=difficulty)
                    success = result.get("status_code") == 200
                    method = agent.current_strategy.get("method", "basic_get")

            except SystemExit:
                logger.warning("Agent %s exited (pruned). Skipping remaining tasks for this agent.", agent_key)
                break
            except Exception as e:
                logger.warning("Task %s failed with exception: %s", task["id"], e)
                success = False

            elapsed_ms = (time.time() - t0) * 1000

            # Update tracking structures
            agent_domain_counts[agent_key][domain] += 1
            method_counts[(domain, method)] += 1

            # Poll eval metrics (non-blocking; fall back to zeros)
            eval_snap = _fetch_eval_metrics(metrics_host, metrics_port)

            spec_idx = _specialisation_index(agent_domain_counts)
            entropy = _shannon_entropy(method_counts)

            record = {
                "task_id": task["id"],
                "condition": condition,
                "domain": domain,
                "difficulty": round(difficulty, 3),
                "task_seq": seq,
                "success": success,
                "agent_id": agent.id,
                "strategy_method": method,
                "elapsed_ms": round(elapsed_ms, 2),
                "soil_trails": eval_snap.get("soil_trails", 0),
                "trails_per_domain": eval_snap.get("trails_per_domain", {}),
                "specialisation_index": round(spec_idx, 4),
                "entropy": round(entropy, 4),
                "inferred_domain": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            fh.write(json.dumps(record) + "\n")
            fh.flush()
            records.append(record)

            if seq % 50 == 0:
                logger.info(
                    "[%s] seq=%d success=%s spec_idx=%.3f entropy=%.3f trails=%d",
                    condition, seq, success, spec_idx, entropy,
                    eval_snap.get("soil_trails", 0),
                )

            _jitter_sleep(_rng)

    logger.info("Condition=%s done. Records written to %s", condition, out_path)
    return records


def run_generic_condition(
    tasks: list[dict],
    n: int,
    seed: int,
    out_path: Path,
    metrics_host: str,
    metrics_port: int,
) -> list[dict]:
    """
    Run the generic-agent condition: cross-domain agents with stigmergy on.
    Used to measure AC2' (specialisation grows from soil, not hard-wired).
    """
    os.environ["EXPLOIT_DISABLED"] = "false"
    os.environ["OG_SEED"] = str(seed)

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("base.") or mod_name.startswith("specialists."):
            del sys.modules[mod_name]

    from specialists.generic import GenericAgent

    agent = GenericAgent(agent_id=f"generic-{seed}")

    agent_domain_counts: dict[str, dict[str, int]] = {
        "generic": defaultdict(int)
    }
    method_counts: Counter = Counter()
    records: list[dict] = []

    import random as _rand
    rng = _rand.Random(seed + 9999)
    selected = tasks[:n]
    rng.shuffle(selected)

    logger.info("Condition=generic seed=%d tasks=%d", seed, len(selected))

    with out_path.open("w") as fh:
        for seq, task in enumerate(selected):
            t0 = time.time()
            try:
                result = agent.execute(task)
                success = bool(result.get("success", False))
                method = result.get("method", "dispatch")
                inferred = agent._inferred_domain or task.get("domain", "unknown")
            except Exception as e:
                logger.warning("Generic task %s failed: %s", task["id"], e)
                success = False
                method = "dispatch"
                inferred = task.get("domain", "unknown")

            elapsed_ms = (time.time() - t0) * 1000

            agent_domain_counts["generic"][inferred] += 1
            method_counts[(inferred, method)] += 1

            eval_snap = _fetch_eval_metrics(metrics_host, metrics_port)
            spec_idx = _specialisation_index(agent_domain_counts)
            entropy = _shannon_entropy(method_counts)

            record = {
                "task_id": task["id"],
                "condition": "generic",
                "domain": task.get("domain", "unknown"),
                "difficulty": float(task.get("difficulty", 0.0)),
                "task_seq": seq,
                "success": success,
                "agent_id": agent.id,
                "strategy_method": method,
                "elapsed_ms": round(elapsed_ms, 2),
                "soil_trails": eval_snap.get("soil_trails", 0),
                "trails_per_domain": eval_snap.get("trails_per_domain", {}),
                "specialisation_index": round(spec_idx, 4),
                "entropy": round(entropy, 4),
                "inferred_domain": inferred,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            records.append(record)

            if seq % 50 == 0:
                logger.info(
                    "[generic] seq=%d success=%s spec_idx=%.3f inferred=%s",
                    seq, success, spec_idx, inferred,
                )

            # Small jitter between tasks
            _jitter_sleep(rng)

    logger.info("Condition=generic done. Records written to %s", out_path)
    return records


def _jitter_sleep(rng) -> None:
    """Sleep 20-80ms with Gaussian jitter to add real stochasticity."""
    try:
        import numpy as np
        delay = max(0.02, np.random.normal(0.05, 0.01))
    except ImportError:
        delay = max(0.02, rng.gauss(0.05, 0.01))
    time.sleep(delay)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run OpenGardener evaluation experiment")
    parser.add_argument(
        "--tasks",
        default="data/generated/tasks.jsonl",
        help="Task corpus JSONL (default: data/generated/tasks.jsonl)",
    )
    parser.add_argument("--n", type=int, default=500, help="Tasks per condition per seed (default 500)")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 0,1 (default: OG_SEED env or 0)")
    parser.add_argument("--out-dir", default="eval/runs", help="Output directory (default: eval/runs)")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: 50 tasks, skip LLM")
    parser.add_argument("--metrics-host", default=os.getenv("GARDENER_HTTP_HOST", "localhost"))
    parser.add_argument("--metrics-port", type=int, default=int(os.getenv("GARDENER_HTTP_PORT", "8080")))
    args = parser.parse_args()

    if args.smoke:
        args.n = 50

    # Resolve seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        env_seed = os.getenv("OG_SEED")
        seeds = [int(env_seed) if env_seed else 0]

    # Load tasks
    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        logger.info("Task corpus not found at %s, generating with default params...", tasks_path)
        # Auto-generate if missing
        gen_script = Path(__file__).parent / "generate_tasks.py"
        import subprocess
        subprocess.run(
            [sys.executable, str(gen_script), "--n", str(max(args.n * 2, 1200)), "--out", str(tasks_path)],
            check=True,
        )

    tasks = []
    with tasks_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))

    logger.info("Loaded %d tasks from %s", len(tasks), tasks_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for seed in seeds:
        import random
        rng = random.Random(seed)
        shuffled = tasks[:]
        rng.shuffle(shuffled)

        for condition, exploit_disabled in [("control", True), ("treatment", False)]:
            out_path = out_dir / f"{condition}_{seed}.jsonl"
            records = run_condition(
                condition=condition,
                tasks=shuffled,
                n=args.n,
                seed=seed,
                out_path=out_path,
                metrics_host=args.metrics_host,
                metrics_port=args.metrics_port,
                exploit_disabled=exploit_disabled,
            )
            all_records.extend(records)

        # Generic condition (cross-domain agents, stigmergy on)
        generic_path = out_dir / f"generic_{seed}.jsonl"
        generic_records = run_generic_condition(
            tasks=shuffled,
            n=args.n,
            seed=seed,
            out_path=generic_path,
            metrics_host=args.metrics_host,
            metrics_port=args.metrics_port,
        )
        all_records.extend(generic_records)

    logger.info("All conditions complete. %d total records. Run eval/analyse.py to generate report.", len(all_records))


if __name__ == "__main__":
    main()
