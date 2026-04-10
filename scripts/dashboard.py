"""
OpenGardener Live Dashboard
───────────────────────────
Polls the Gardener metrics endpoint and ChromaDB to display a live
terminal view of the ecosystem: agents, soil trails, and system health.

Usage:
  python dashboard.py                    # defaults: gardener=localhost:8080, chroma=localhost:8000
  GARDENER_HOST=gardener CHROMA_HOST=chromadb python dashboard.py

Inside Docker:
  make dashboard
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

GARDENER_HOST = os.getenv("GARDENER_HOST", "localhost")
GARDENER_PORT = os.getenv("METRICS_PORT", "8080")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
REFRESH_SECS = float(os.getenv("DASHBOARD_REFRESH_S", "3"))

GARDENER_METRICS_URL = f"http://{GARDENER_HOST}:{GARDENER_PORT}/metrics"
CHROMA_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2"
COLLECTION_NAME = "opengardener_soil"

console = Console()


def fetch_metrics() -> Optional[dict]:
    try:
        r = httpx.get(GARDENER_METRICS_URL, timeout=3.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_soil_breakdown() -> dict:
    """Fetch per-domain trail counts from ChromaDB."""
    try:
        # Get all trail metadata (limit to recent 1000)
        r = httpx.post(
            f"{CHROMA_URL}/collections/{COLLECTION_NAME}/query",
            json={
                "query_embeddings": [[0.0] * 384],
                "n_results": 500,
                "include": ["metadatas", "distances"],
            },
            timeout=5.0,
        )
        if r.status_code != 200:
            return {}

        data = r.json()
        metadatas = data.get("metadatas", [[]])[0]

        breakdown: dict[str, dict] = {}
        for meta in metadatas:
            domain = meta.get("task_domain", "unknown")
            outcome = meta.get("outcome", "unknown")
            if domain not in breakdown:
                breakdown[domain] = {"success": 0, "failure": 0, "total": 0}
            breakdown[domain][outcome] = breakdown[domain].get(outcome, 0) + 1
            breakdown[domain]["total"] += 1

        return breakdown
    except Exception:
        return {}


def status_color(status: str) -> str:
    return {
        "active": "green",
        "struggling": "yellow",
        "idle": "cyan",
        "unknown": "dim",
    }.get(status, "white")


def failure_rate_color(rate: float) -> str:
    if rate >= 0.7:
        return "red"
    if rate >= 0.4:
        return "yellow"
    return "green"


def format_age(age_secs: int) -> str:
    if age_secs < 60:
        return f"{age_secs}s"
    if age_secs < 3600:
        return f"{age_secs // 60}m"
    return f"{age_secs // 3600}h {(age_secs % 3600) // 60}m"


def build_agent_table(agents: list) -> Table:
    table = Table(
        title="Active Agents",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Agent ID", style="dim", width=18)
    table.add_column("Type", width=16)
    table.add_column("Domain", width=16)
    table.add_column("Tasks", justify="right", width=7)
    table.add_column("Fail%", justify="right", width=7)
    table.add_column("Status", width=11)
    table.add_column("Age", justify="right", width=8)

    if not agents:
        table.add_row("[dim]No agents registered yet[/dim]", "", "", "", "", "", "")
        return table

    for a in sorted(agents, key=lambda x: x["tasks_completed"], reverse=True):
        agent_id = a["id"][:16]
        fail_rate = a["failure_rate"]
        status = a["status"]
        color = failure_rate_color(fail_rate)
        status_col = f"[{status_color(status)}]{status}[/{status_color(status)}]"
        table.add_row(
            agent_id,
            a["agent_type"],
            a["domain"],
            str(a["tasks_completed"]),
            f"[{color}]{fail_rate * 100:.0f}%[/{color}]",
            status_col,
            format_age(a["age_secs"]),
        )

    return table


def build_soil_table(breakdown: dict, total_trails: int) -> Table:
    table = Table(
        title=f"Soil Trails  (total: {total_trails})",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("Domain", width=18)
    table.add_column("Total", justify="right", width=7)
    table.add_column("Success", justify="right", width=8)
    table.add_column("Failure", justify="right", width=8)
    table.add_column("Success%", justify="right", width=9)

    if not breakdown:
        table.add_row("[dim]No trails yet — run: make seed[/dim]", "", "", "", "")
        return table

    for domain, counts in sorted(breakdown.items(), key=lambda x: x[1]["total"], reverse=True):
        total = counts["total"]
        success = counts.get("success", 0)
        failure = counts.get("failure", 0)
        pct = (success / total * 100) if total > 0 else 0
        pct_color = "green" if pct >= 60 else ("yellow" if pct >= 40 else "red")
        table.add_row(
            domain,
            str(total),
            f"[green]{success}[/green]",
            f"[red]{failure}[/red]",
            f"[{pct_color}]{pct:.0f}%[/{pct_color}]",
        )

    return table


def build_summary_panel(metrics: Optional[dict], soil_breakdown: dict) -> Panel:
    if metrics is None:
        content = Text("Waiting for Gardener...", style="yellow")
        return Panel(content, title="System", border_style="yellow")

    agents = metrics.get("agents", [])
    total_agents = len(agents)
    struggling = sum(1 for a in agents if a["status"] == "struggling")
    total_tasks = sum(a["tasks_completed"] for a in agents)
    total_trails = metrics["soil"]["total_trails"]
    domains = len(soil_breakdown)

    ts = metrics.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_str = dt.strftime("%H:%M:%S UTC")
    except Exception:
        ts_str = ts

    lines = [
        f"[bold]Agents:[/bold]  {total_agents} active  |  [yellow]{struggling} struggling[/yellow]",
        f"[bold]Tasks:[/bold]   {total_tasks} completed across all agents",
        f"[bold]Soil:[/bold]    {total_trails} trails  |  {domains} domains",
        f"[dim]Updated: {ts_str}  •  refresh every {REFRESH_SECS:.0f}s[/dim]",
    ]

    content = Text.from_markup("\n".join(lines))
    return Panel(content, title="[bold green]OpenGardener[/bold green]", border_style="green")


def build_layout(metrics: Optional[dict], soil_breakdown: dict) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="agents"),
        Layout(name="soil"),
    )

    layout["header"].update(build_summary_panel(metrics, soil_breakdown))

    if metrics:
        agents = metrics.get("agents", [])
        layout["agents"].update(build_agent_table(agents))
        layout["soil"].update(
            build_soil_table(soil_breakdown, metrics["soil"]["total_trails"])
        )
    else:
        layout["agents"].update(Panel("[yellow]Connecting to Gardener...[/yellow]", title="Agents"))
        layout["soil"].update(Panel("[yellow]Connecting to ChromaDB...[/yellow]", title="Soil"))

    return layout


def run():
    console.print(
        "\n[bold green]OpenGardener Dashboard[/bold green]  "
        f"[dim]gardener={GARDENER_HOST}:{GARDENER_PORT}  chroma={CHROMA_HOST}:{CHROMA_PORT}[/dim]\n"
        "[dim]Press Ctrl+C to exit[/dim]\n"
    )

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            metrics = fetch_metrics()
            soil_breakdown = fetch_soil_breakdown()
            live.update(build_layout(metrics, soil_breakdown))
            time.sleep(REFRESH_SECS)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard closed.[/dim]")
        sys.exit(0)
