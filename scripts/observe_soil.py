"""
Observe Soil Script — queries ChromaDB to display pheromone trail statistics.

Shows:
  - Total trail count
  - Trails per domain (data_cleaning, code_generation, api_testing)
  - Top approaches per domain (most-used successful strategies)
  - Failure markers (what to avoid)
  - Agent reputation summary

Usage:
  make observe          # via Docker Compose
  python scripts/observe_soil.py  # locally
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from collections import defaultdict

logging.basicConfig(level="INFO", format="%(message)s")
logger = logging.getLogger(__name__)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "opengardener_soil"


def get_chroma_client():
    import chromadb
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return client


def print_header(title: str):
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(title: str):
    print(f"\n── {title} {'─' * (50 - len(title))}")


def observe():
    print_header("OpenGardener Soil Observer")
    print(f"  ChromaDB: {CHROMA_HOST}:{CHROMA_PORT}")
    print(f"  Time:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    try:
        client = get_chroma_client()
    except Exception as e:
        print(f"\n  ERROR: Cannot connect to ChromaDB — {e}")
        print(f"  Is ChromaDB running? Try: docker compose up chromadb")
        return

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"\n  No soil collection found ('{COLLECTION_NAME}' doesn't exist yet).")
        print("  Run 'make seed' to initialize the soil.")
        return

    # Get all trails
    count = collection.count()
    print(f"\n  Total pheromone trails: {count}")

    if count == 0:
        print("\n  Soil is empty. Run 'make seed' to populate it.")
        return

    # Fetch all records (V1 — small enough to fit in memory)
    results = collection.get(include=["metadatas", "documents"], limit=10000)
    metadatas = results["metadatas"]

    # ── Domain breakdown ──────────────────────────────────────
    print_section("Trail Distribution by Domain")

    domain_stats = defaultdict(lambda: {"success": 0, "failure": 0, "agents": set()})
    approach_counts = defaultdict(lambda: defaultdict(int))

    for meta in metadatas:
        domain = meta.get("task_domain", "unknown")
        outcome = meta.get("outcome", "unknown")
        agent_id = meta.get("agent_id", "unknown")
        approach_raw = meta.get("approach", "{}")

        domain_stats[domain][outcome] = domain_stats[domain].get(outcome, 0) + 1
        domain_stats[domain]["agents"].add(agent_id)

        # Count approach methods
        try:
            approach = json.loads(approach_raw) if isinstance(approach_raw, str) else approach_raw
            method = approach.get("method", "unknown") if isinstance(approach, dict) else "unknown"
        except (json.JSONDecodeError, AttributeError):
            method = "unknown"

        if outcome == "success":
            approach_counts[domain][method] += 1

    domains = ["data_cleaning", "code_generation", "api_testing"]
    for domain in domains:
        stats = domain_stats.get(domain, {"success": 0, "failure": 0, "agents": set()})
        total = stats["success"] + stats.get("failure", 0)
        success_rate = stats["success"] / total if total > 0 else 0
        agent_count = len(stats["agents"])

        print(f"  {domain:<22} total={total:>4}  success_rate={success_rate:.0%}  agents={agent_count}")

    # ── Top approaches per domain ──────────────────────────────
    print_section("Top Successful Approaches by Domain")

    for domain in domains:
        methods = approach_counts.get(domain, {})
        if not methods:
            print(f"  {domain}: no successful trails yet")
            continue

        top = sorted(methods.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {domain}:")
        for method, count in top:
            print(f"    [{count:>3}x] {method}")

    # ── Failure analysis ──────────────────────────────────────
    print_section("Failure Markers (approaches to avoid)")

    failure_metas = [m for m in metadatas if m.get("outcome") == "failure"]
    if not failure_metas:
        print("  No failure markers in soil (all sunny skies!)")
    else:
        failure_domains = defaultdict(list)
        for m in failure_metas:
            domain = m.get("task_domain", "unknown")
            severity = m.get("severity", "unknown")
            error_summary = m.get("task_summary", "")[:60]
            failure_domains[domain].append(f"[{severity}] {error_summary}")

        for domain, failures in failure_domains.items():
            print(f"  {domain}: {len(failures)} failures")
            for f in failures[:3]:
                print(f"    • {f}")

    # ── Agent summary ─────────────────────────────────────────
    print_section("Agent Activity Summary")

    agent_stats = defaultdict(lambda: {"success": 0, "failure": 0})
    for meta in metadatas:
        agent_id = meta.get("agent_id", "unknown")
        outcome = meta.get("outcome", "unknown")
        agent_stats[agent_id][outcome] = agent_stats[agent_id].get(outcome, 0) + 1

    if not agent_stats:
        print("  No agent activity recorded.")
    else:
        print(f"  {'Agent ID':<30} {'Tasks':>6}  {'Success%':>9}")
        print(f"  {'─'*30}  {'─'*6}  {'─'*9}")
        for agent_id, stats in sorted(agent_stats.items()):
            total = stats["success"] + stats.get("failure", 0)
            rate = stats["success"] / total if total > 0 else 0
            print(f"  {agent_id:<30} {total:>6}  {rate:>8.0%}")

    print("\n" + "═" * 60)
    print("  Run 'make observe' again after more tasks to see evolution.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    observe()
