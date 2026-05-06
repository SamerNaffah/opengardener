"""
OpenGardener evaluation analysis — Round 2.

Reads all JSONL run files from eval/runs/ (or --runs-dir), computes the five
Round-2 acceptance criteria, generates matplotlib plots, and writes eval/RESULTS.md.

Acceptance criteria (all must pass):
  AC1'  On tasks with difficulty > 0.5: treatment_SR − control_SR ≥ 0.20
        Soil retrieval helps when the task is non-trivial.
  AC2'  GenericAgent mean specialisation index at task 500 ≥ 0.6
        Agents learn to specialise when they had a choice.
  AC2'b GenericAgent mean specialisation index at task 100 ≤ 0.5
        Specialisation isn't immediate — it grows from experience.
  AC3'  Per-domain approach entropy: 0.5 ≤ H ≤ 2.5 bits for all domains
        Bounded diversity (not collapsed, not chaos).
  AC4'  Per-task SR variance across two seeds > 0.02 (paths differ)
        AND |final_sr_seed_a − final_sr_seed_b| < 0.05 (destinations match)
        Real stochasticity converging to consistent outcomes.

Plots (saved to eval/plots/):
  01_success_rate.png        — cumulative success rate over tasks, control vs treatment
  02_specialisation.png      — mean specialisation index over tasks
  03_entropy.png             — Shannon entropy over tasks
  04_trails_per_domain.png   — pheromone trails accumulated per domain

Usage:
  python eval/analyse.py
  python eval/analyse.py --runs-dir eval/runs --out eval/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ─── Data loading ────────────────────────────────────────────────────────────

class RunRecord(NamedTuple):
    task_id: str
    condition: str
    domain: str
    task_seq: int
    success: bool
    agent_id: str
    strategy_method: str
    elapsed_ms: float
    soil_trails: int
    trails_per_domain: dict
    specialisation_index: float
    entropy: float
    timestamp: str
    seed: int
    difficulty: float
    inferred_domain: str  # non-empty only for generic condition


def _load_runs(runs_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in sorted(runs_dir.glob("*.jsonl")):
        # filename: {condition}_{seed}.jsonl
        stem = path.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        condition, seed_str = parts
        try:
            seed = int(seed_str)
        except ValueError:
            continue

        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                records.append(RunRecord(
                    task_id=d.get("task_id", ""),
                    condition=d.get("condition", condition),
                    domain=d.get("domain", ""),
                    task_seq=d.get("task_seq", 0),
                    success=bool(d.get("success", False)),
                    agent_id=d.get("agent_id", ""),
                    strategy_method=d.get("strategy_method", ""),
                    elapsed_ms=float(d.get("elapsed_ms", 0.0)),
                    soil_trails=int(d.get("soil_trails", 0)),
                    trails_per_domain=d.get("trails_per_domain", {}),
                    specialisation_index=float(d.get("specialisation_index", 0.0)),
                    entropy=float(d.get("entropy", 0.0)),
                    timestamp=d.get("timestamp", ""),
                    seed=seed,
                    difficulty=float(d.get("difficulty", 0.0)),
                    inferred_domain=d.get("inferred_domain") or "",
                ))
    return records


# ─── Metric helpers ──────────────────────────────────────────────────────────

def _cumulative_success_rate(records: list[RunRecord]) -> list[float]:
    hits = 0
    rates = []
    for i, r in enumerate(records, 1):
        if r.success:
            hits += 1
        rates.append(hits / i)
    return rates


def _rolling(vals: list[float], window: int = 20) -> list[float]:
    out = []
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        out.append(sum(vals[start:i+1]) / (i - start + 1))
    return out


def _group_by(records: list[RunRecord], key) -> dict[str, list[RunRecord]]:
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return dict(groups)


# ─── Plots ───────────────────────────────────────────────────────────────────

_COLORS = {"control": "#6b7280", "treatment": "#16a34a"}
_STYLE = {"linewidth": 1.8, "alpha": 0.9}


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_success_rate(
    by_condition: dict[str, list[RunRecord]],
    plots_dir: Path,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(10, 5))
    final_rates: dict[str, float] = {}

    for cond, recs in sorted(by_condition.items()):
        recs_sorted = sorted(recs, key=lambda r: r.task_seq)
        rates = _cumulative_success_rate(recs_sorted)
        xs = list(range(1, len(rates) + 1))
        ax.plot(xs, rates, label=cond, color=_COLORS.get(cond, "steelblue"), **_STYLE)
        final_rates[cond] = rates[-1] if rates else 0.0

    ax.set_xlabel("Tasks completed")
    ax.set_ylabel("Cumulative success rate")
    ax.set_title("Cumulative success rate: control vs. treatment")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, plots_dir / "01_success_rate.png")
    return final_rates


def plot_specialisation(
    by_condition: dict[str, list[RunRecord]],
    plots_dir: Path,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_at_500: dict[str, float] = {}

    for cond, recs in sorted(by_condition.items()):
        recs_sorted = sorted(recs, key=lambda r: r.task_seq)
        spec_vals = [r.specialisation_index for r in recs_sorted]
        rolled = _rolling(spec_vals, window=30)
        xs = list(range(1, len(rolled) + 1))
        ax.plot(xs, rolled, label=cond, color=_COLORS.get(cond, "steelblue"), **_STYLE)
        # Mean across all tasks up to task 500
        slice_500 = spec_vals[:500]
        mean_at_500[cond] = float(np.mean(slice_500)) if slice_500 else 0.0

    ax.axhline(0.6, color="crimson", linestyle="--", linewidth=1.2, label="AC2 threshold (0.6)")
    ax.set_xlabel("Tasks completed")
    ax.set_ylabel("Mean specialisation index (rolling 30)")
    ax.set_title("Agent specialisation over time")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, plots_dir / "02_specialisation.png")
    return mean_at_500


def plot_entropy(
    by_condition: dict[str, list[RunRecord]],
    plots_dir: Path,
) -> dict[str, list[float]]:
    fig, ax = plt.subplots(figsize=(10, 5))
    entropy_series: dict[str, list[float]] = {}

    for cond, recs in sorted(by_condition.items()):
        recs_sorted = sorted(recs, key=lambda r: r.task_seq)
        ent_vals = [r.entropy for r in recs_sorted]
        rolled = _rolling(ent_vals, window=30)
        xs = list(range(1, len(rolled) + 1))
        ax.plot(xs, rolled, label=cond, color=_COLORS.get(cond, "steelblue"), **_STYLE)
        entropy_series[cond] = ent_vals

    ax.axhline(0.5, color="orange", linestyle="--", linewidth=1.1, label="AC3 lower bound (0.5)")
    ax.axhline(3.5, color="orange", linestyle=":",  linewidth=1.1, label="AC3 upper bound (3.5)")
    ax.set_xlabel("Tasks completed")
    ax.set_ylabel("Shannon entropy H(domain,method) bits (rolling 30)")
    ax.set_title("Strategy entropy over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, plots_dir / "03_entropy.png")
    return entropy_series


def plot_trails(
    by_condition: dict[str, list[RunRecord]],
    plots_dir: Path,
):
    domains = ["data_cleaning", "code_generation", "api_testing"]
    fig, axes = plt.subplots(1, len(domains), figsize=(14, 4), sharey=False)

    for ax, domain in zip(axes, domains):
        for cond, recs in sorted(by_condition.items()):
            recs_sorted = sorted(recs, key=lambda r: r.task_seq)
            trail_vals = [r.trails_per_domain.get(domain, 0) for r in recs_sorted]
            xs = list(range(1, len(trail_vals) + 1))
            ax.plot(xs, trail_vals, label=cond, color=_COLORS.get(cond, "steelblue"), **_STYLE)
        ax.set_title(domain.replace("_", " ").title())
        ax.set_xlabel("Tasks")
        ax.set_ylabel("Trails in domain")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Pheromone trails per domain", fontsize=12)
    fig.tight_layout()
    _save(fig, plots_dir / "04_trails_per_domain.png")


# ─── Acceptance criteria (Round 2) ───────────────────────────────────────────

def _evaluate_criteria(
    final_rates: dict[str, float],
    mean_spec_at_500: dict[str, float],
    entropy_series: dict[str, list[float]],
    seeds: list[int],
    by_condition_seed: dict[tuple[str, int], list[RunRecord]],
    all_records: list[RunRecord],
) -> list[dict]:
    results = []

    # ── AC1': On hard tasks (difficulty > 0.5), treatment_SR − control_SR ≥ 0.20
    hard_ctrl = [r for r in all_records if r.condition == "control" and r.difficulty > 0.5]
    hard_trt  = [r for r in all_records if r.condition == "treatment" and r.difficulty > 0.5]
    if hard_ctrl and hard_trt:
        ctrl_hard_sr = sum(1 for r in hard_ctrl if r.success) / len(hard_ctrl)
        trt_hard_sr  = sum(1 for r in hard_trt  if r.success) / len(hard_trt)
        delta = trt_hard_sr - ctrl_hard_sr
        results.append({
            "id": "AC1'",
            "description": "Hard-task (difficulty>0.5) treatment SR − control SR ≥ 0.20",
            "value": f"Δ = {delta:.3f}  (treatment={trt_hard_sr:.3f}, control={ctrl_hard_sr:.3f}, n={len(hard_trt)})",
            "pass": delta >= 0.20,
        })
    else:
        results.append({
            "id": "AC1'",
            "description": "Hard-task SR delta ≥ 0.20",
            "value": f"insufficient hard-task records (ctrl={len(hard_ctrl)}, trt={len(hard_trt)})",
            "pass": None,
        })

    # ── AC2': GenericAgent spec_idx at task 500 ≥ 0.6
    generic_recs = sorted(
        [r for r in all_records if r.condition == "generic"],
        key=lambda r: r.task_seq,
    )
    if generic_recs:
        at_500 = [r for r in generic_recs if r.task_seq <= 500]
        spec_at_500 = at_500[-1].specialisation_index if at_500 else 0.0
        results.append({
            "id": "AC2'",
            "description": "GenericAgent specialisation index at task 500 ≥ 0.6",
            "value": f"spec_idx@500 = {spec_at_500:.4f}",
            "pass": spec_at_500 >= 0.6,
        })

        # ── AC2'b: GenericAgent spec_idx at task 100 ≤ 0.5 (gradual, not instant)
        at_100 = [r for r in generic_recs if r.task_seq <= 100]
        spec_at_100 = at_100[-1].specialisation_index if at_100 else 0.0
        results.append({
            "id": "AC2'b",
            "description": "GenericAgent specialisation index at task 100 ≤ 0.5 (gradual growth)",
            "value": f"spec_idx@100 = {spec_at_100:.4f}",
            "pass": spec_at_100 <= 0.5,
        })
    else:
        results.append({"id": "AC2'",  "description": "GenericAgent spec@500 ≥ 0.6", "value": "no generic records", "pass": None})
        results.append({"id": "AC2'b", "description": "GenericAgent spec@100 ≤ 0.5", "value": "no generic records", "pass": None})

    # ── AC3': Per-domain entropy 0.5 ≤ H ≤ 2.5 bits (tighter upper bound than R1)
    trt_recs = [r for r in all_records if r.condition == "treatment"]
    domains = sorted({r.domain for r in trt_recs})
    domain_ent_ok = True
    domain_ent_vals = {}
    for dom in domains:
        dom_recs = [r for r in trt_recs if r.domain == dom]
        method_counts: dict[str, int] = {}
        for r in dom_recs:
            method_counts[r.strategy_method] = method_counts.get(r.strategy_method, 0) + 1
        total = sum(method_counts.values())
        if total == 0:
            h = 0.0
        else:
            h = -sum((c / total) * math.log2(c / total) for c in method_counts.values() if c > 0)
        domain_ent_vals[dom] = h
        if not (0.5 <= h <= 2.5):
            domain_ent_ok = False
    ent_summary = "  ".join(f"{d}={v:.2f}" for d, v in domain_ent_vals.items())
    results.append({
        "id": "AC3'",
        "description": "Per-domain entropy in [0.5, 2.5] bits for all domains",
        "value": ent_summary or "no data",
        "pass": domain_ent_ok if domain_ent_vals else None,
    })

    # ── AC4': per-task SR variance > 0.02 (paths differ) AND |Δ final SR| < 0.05
    if len(seeds) >= 2:
        s0, s1 = seeds[0], seeds[1]
        recs_s0 = sorted(by_condition_seed.get(("treatment", s0), []), key=lambda r: r.task_seq)
        recs_s1 = sorted(by_condition_seed.get(("treatment", s1), []), key=lambda r: r.task_seq)
        # Align by task_seq up to min length
        min_len = min(len(recs_s0), len(recs_s1))
        if min_len > 10:
            successes_s0 = [float(r.success) for r in recs_s0[:min_len]]
            successes_s1 = [float(r.success) for r in recs_s1[:min_len]]
            # Per-task variance: variance of (s0[i] XOR s1[i]) ≈ variance of abs diff
            per_task_diff = [abs(a - b) for a, b in zip(successes_s0, successes_s1)]
            variance = float(np.var(per_task_diff))
            final_sr_s0 = sum(successes_s0) / len(successes_s0)
            final_sr_s1 = sum(successes_s1) / len(successes_s1)
            final_delta = abs(final_sr_s0 - final_sr_s1)
            paths_differ = variance > 0.02
            destinations_match = final_delta < 0.05
            results.append({
                "id": "AC4'",
                "description": "Per-task variance > 0.02 (paths differ) AND |Δ final SR| < 0.05",
                "value": f"variance={variance:.4f}  |Δ_final|={final_delta:.4f}",
                "pass": paths_differ and destinations_match,
            })
        else:
            results.append({"id": "AC4'", "description": "Variance > 0.02 AND Δ < 0.05", "value": "insufficient records per seed", "pass": None})
    else:
        results.append({"id": "AC4'", "description": "Variance > 0.02 AND Δ < 0.05", "value": "only one seed — skipped", "pass": None})

    return results


# ─── Report writer ───────────────────────────────────────────────────────────

def write_results_md(
    out_path: Path,
    criteria: list[dict],
    final_rates: dict[str, float],
    mean_spec_at_500: dict[str, float],
    entropy_series: dict[str, list[float]],
    total_records: int,
    seeds: list[int],
    plots_dir: Path,
    out_dir: Path,
):
    passed = [c for c in criteria if c["pass"] is True]
    failed = [c for c in criteria if c["pass"] is False]
    skipped = [c for c in criteria if c["pass"] is None]
    overall = len(failed) == 0

    lines = []
    lines.append("# OpenGardener — Evaluation Results")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Seeds:** {seeds}")
    lines.append(f"**Total task records:** {total_records}")
    lines.append(f"**Overall verdict:** {'✅ PASS' if overall else '❌ FAIL'} ({len(passed)}/{len(criteria) - len(skipped)} criteria met)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Acceptance Criteria")
    lines.append("")
    lines.append("| ID | Description | Value | Result |")
    lines.append("|----|-------------|-------|--------|")
    for c in criteria:
        icon = "✅" if c["pass"] is True else ("❌" if c["pass"] is False else "⚠️ skip")
        lines.append(f"| {c['id']} | {c['description']} | `{c['value']}` | {icon} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Key Metrics Summary")
    lines.append("")
    lines.append("| Metric | Control | Treatment |")
    lines.append("|--------|---------|-----------|")
    lines.append(f"| Final cumulative success rate | {final_rates.get('control', 0):.3f} | {final_rates.get('treatment', 0):.3f} |")
    lines.append(f"| Mean specialisation index (≤500 tasks) | {mean_spec_at_500.get('control', 0):.3f} | {mean_spec_at_500.get('treatment', 0):.3f} |")

    ctrl_ents = entropy_series.get("control", [])
    trt_ents  = entropy_series.get("treatment", [])
    ctrl_ent_final = ctrl_ents[-1] if ctrl_ents else 0.0
    trt_ent_final  = trt_ents[-1]  if trt_ents  else 0.0
    lines.append(f"| Final Shannon entropy (bits) | {ctrl_ent_final:.3f} | {trt_ent_final:.3f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    plots_rel = plots_dir.relative_to(out_path.parent) if plots_dir.is_relative_to(out_path.parent) else plots_dir
    for fname, caption in [
        ("01_success_rate.png", "Cumulative success rate — control vs. treatment"),
        ("02_specialisation.png", "Mean agent specialisation index over time"),
        ("03_entropy.png", "Shannon entropy of (domain, method) over time"),
        ("04_trails_per_domain.png", "Pheromone trails accumulated per domain"),
    ]:
        lines.append(f"### {caption}")
        lines.append(f"![{caption}]({plots_rel}/{fname})")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("**Control condition:** `EXPLOIT_DISABLED=true` — agents always explore (no soil feedback).")
    lines.append("**Treatment condition:** `EXPLOIT_DISABLED=false` — stigmergic specialisation enabled.")
    lines.append("")
    lines.append("**Specialisation index:** fraction of each agent's tasks in its dominant domain, averaged across all agents.")
    lines.append("**Shannon entropy:** `H = -Σ p(domain,method) log₂ p(domain,method)` over the running (domain, method) distribution.")
    lines.append("  Lower bound (0.5) prevents total strategy collapse; upper bound (3.5) confirms learning occurred.")
    lines.append("")
    lines.append("Run files: `eval/runs/{control,treatment}_{seed}.jsonl`")
    lines.append("Re-run: `python scripts/run_experiment.py --n 500 --seeds 0,1 && python eval/analyse.py`")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse OpenGardener experiment results")
    parser.add_argument("--runs-dir", default="eval/runs", help="Directory of JSONL run files")
    parser.add_argument("--out", default="eval/RESULTS.md", help="Output report path")
    parser.add_argument("--plots-dir", default="eval/plots", help="Directory for plots")
    args = parser.parse_args()

    # Resolve paths relative to repo root if invoked from anywhere
    _here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
    repo_root = _here.parent
    runs_dir  = Path(args.runs_dir)  if Path(args.runs_dir).is_absolute()  else repo_root / args.runs_dir
    out_path  = Path(args.out)       if Path(args.out).is_absolute()       else repo_root / args.out
    plots_dir = Path(args.plots_dir) if Path(args.plots_dir).is_absolute() else repo_root / args.plots_dir

    records = _load_runs(runs_dir)
    if not records:
        print(f"No JSONL files found in {runs_dir}. Run scripts/run_experiment.py first.")
        sys.exit(1)

    print(f"Loaded {len(records)} records from {runs_dir}")

    by_condition = _group_by(records, lambda r: r.condition)
    seeds = sorted({r.seed for r in records})
    by_condition_seed: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        by_condition_seed[(r.condition, r.seed)].append(r)

    print("Generating plots...")
    plots_dir.mkdir(parents=True, exist_ok=True)
    final_rates     = plot_success_rate(by_condition, plots_dir)
    mean_spec_at_500 = plot_specialisation(by_condition, plots_dir)
    entropy_series  = plot_entropy(by_condition, plots_dir)
    plot_trails(by_condition, plots_dir)

    print("Evaluating acceptance criteria...")
    criteria = _evaluate_criteria(
        final_rates=final_rates,
        mean_spec_at_500=mean_spec_at_500,
        entropy_series=entropy_series,
        seeds=seeds,
        by_condition_seed=by_condition_seed,
        all_records=records,
    )

    for c in criteria:
        icon = "PASS" if c["pass"] is True else ("FAIL" if c["pass"] is False else "SKIP")
        print(f"  [{icon}] {c['id']}: {c['value']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_results_md(
        out_path=out_path,
        criteria=criteria,
        final_rates=final_rates,
        mean_spec_at_500=mean_spec_at_500,
        entropy_series=entropy_series,
        total_records=len(records),
        seeds=seeds,
        plots_dir=plots_dir,
        out_dir=runs_dir,
    )

    failed = [c for c in criteria if c["pass"] is False]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
