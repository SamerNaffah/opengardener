"""
OpenGardener evaluation analysis.

Reads all JSONL run files from eval/runs/ (or --runs-dir), computes the four
acceptance criteria, generates matplotlib plots, and writes eval/RESULTS.md.

Acceptance criteria (all must pass):
  AC1  Cumulative success rate treatment >= 1.3× baseline (control)
  AC2  Mean specialisation index >= 0.6 within 500 tasks
  AC3  Shannon entropy of (domain, method) stays bounded (0.5 <= H <= 3.5)
       — lower bound prevents total collapse, upper prevents no learning
  AC4  Results reproducible across two seeds (|treatment_sr_seed0 - treatment_sr_seed1| < 0.05)

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


# ─── Acceptance criteria ─────────────────────────────────────────────────────

def _evaluate_criteria(
    final_rates: dict[str, float],
    mean_spec_at_500: dict[str, float],
    entropy_series: dict[str, list[float]],
    seeds: list[int],
    by_condition_seed: dict[tuple[str, int], list[RunRecord]],
) -> list[dict]:
    results = []

    # AC1: treatment SR >= 1.3 × control SR
    ctrl_sr = final_rates.get("control", 0.0)
    trt_sr  = final_rates.get("treatment", 0.0)
    ratio = (trt_sr / ctrl_sr) if ctrl_sr > 0 else 0.0
    results.append({
        "id": "AC1",
        "description": "Cumulative success rate treatment ≥ 1.3× control",
        "value": f"ratio = {ratio:.3f}  (treatment={trt_sr:.3f}, control={ctrl_sr:.3f})",
        "pass": ratio >= 1.3,
    })

    # AC2: mean specialisation index >= 0.6 within 500 tasks
    trt_spec = mean_spec_at_500.get("treatment", 0.0)
    results.append({
        "id": "AC2",
        "description": "Mean specialisation index ≥ 0.6 within 500 tasks",
        "value": f"mean_spec_index = {trt_spec:.4f}",
        "pass": trt_spec >= 0.6,
    })

    # AC3: entropy bounded in [0.5, 3.5] for treatment
    trt_ents = entropy_series.get("treatment", [])
    if trt_ents:
        final_ent = trt_ents[-1]
        ent_ok = 0.5 <= final_ent <= 3.5
    else:
        final_ent = 0.0
        ent_ok = False
    results.append({
        "id": "AC3",
        "description": "Final Shannon entropy H in [0.5, 3.5] bits",
        "value": f"H = {final_ent:.4f} bits",
        "pass": ent_ok,
    })

    # AC4: reproducible across two seeds (|sr_seed0 - sr_seed1| < 0.05)
    if len(seeds) >= 2:
        sr_by_seed = {}
        for s in seeds[:2]:
            recs = by_condition_seed.get(("treatment", s), [])
            if recs:
                rates = _cumulative_success_rate(sorted(recs, key=lambda r: r.task_seq))
                sr_by_seed[s] = rates[-1]
        if len(sr_by_seed) == 2:
            s0, s1 = seeds[0], seeds[1]
            diff = abs(sr_by_seed.get(s0, 0) - sr_by_seed.get(s1, 0))
            results.append({
                "id": "AC4",
                "description": "Reproducible across two seeds: |Δ success_rate| < 0.05",
                "value": f"|seed{s0}={sr_by_seed.get(s0,0):.3f} − seed{s1}={sr_by_seed.get(s1,0):.3f}| = {diff:.4f}",
                "pass": diff < 0.05,
            })
        else:
            results.append({
                "id": "AC4",
                "description": "Reproducible across two seeds",
                "value": "only one seed available — skipped",
                "pass": None,
            })
    else:
        results.append({
            "id": "AC4",
            "description": "Reproducible across two seeds",
            "value": "only one seed run — skipped",
            "pass": None,
        })

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
