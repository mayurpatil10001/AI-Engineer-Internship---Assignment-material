"""
CLI entry point for the OrbitDesk Support Agent.

Usage:
    python -m src.cli ask "Your question here"
    python -m src.cli ask "Your question here" --json
    python -m src.cli ask "Your question here" --timing

The CLI:
  - Loads the compiled graph
  - Runs the query through the full agent pipeline
  - Prints a formatted response with node trace, sources, and timing
  - Optionally prints raw JSON output
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Add project root to path (allows `python -m src.cli` from any working dir) ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.agent.logging_config import setup_logging, dump_run_summary, get_run_timestamp
from src.agent.state import make_initial_state
from src.agent.models import get_timing_report

console = Console()


def run_query(question: str) -> dict:
    """
    Run a question through the compiled support agent graph.

    Args:
        question: The user's question string.

    Returns:
        The final AgentState as a dict.
    """
    from src.agent.graph import get_compiled_app

    app = get_compiled_app()
    initial_state = make_initial_state(question)

    t0 = time.perf_counter()
    final_state = app.invoke(initial_state)
    wall_clock_ms = (time.perf_counter() - t0) * 1000

    run_id = get_run_timestamp()
    dump_run_summary(run_id, question, final_state, wall_clock_ms)

    return final_state


def format_response(state: dict, show_timing: bool = False) -> None:
    """Print a rich-formatted response to stdout."""

    classification = state.get("classification", "unknown")
    final_answer = state.get("final_answer", "(no answer)")
    confidence = state.get("confidence", 0.0)
    requires_human = state.get("requires_human", False)
    reason = state.get("reason", "")
    node_trace = state.get("node_trace", [])
    sources = state.get("sources", [])
    warnings = state.get("warnings", [])
    clarifying_question = state.get("clarifying_question")
    escalation_reason = state.get("escalation_reason")

    # ── Classification badge ──────────────────────────────────────────────────
    badge_colors = {
        "answerable": "green",
        "requires_clarification": "yellow",
        "requires_escalation": "orange3",
        "out_of_scope": "red",
        "safe_failure": "bold red",
    }
    color = badge_colors.get(classification, "white")
    console.print(f"\n[{color}]▶ {classification.upper().replace('_', ' ')}[/{color}]  "
                  f"confidence={confidence:.2f}  requires_human={requires_human}")

    # ── Answer ────────────────────────────────────────────────────────────────
    console.print(Panel(final_answer, title="Answer", border_style=color, expand=False))

    # ── Clarifying question ───────────────────────────────────────────────────
    if clarifying_question:
        console.print(Panel(clarifying_question, title="Clarifying Question", border_style="yellow"))

    # ── Sources ───────────────────────────────────────────────────────────────
    if sources:
        t = Table(title="Sources", show_header=True, header_style="bold cyan")
        t.add_column("Source ID", style="cyan")
        t.add_column("Excerpt", max_width=80)
        for s in sources:
            src_id = s.get("source_id") if isinstance(s, dict) else s["source_id"]
            passage = s.get("passage") if isinstance(s, dict) else s["passage"]
            t.add_row(src_id, (passage or "")[:200])
        console.print(t)

    # ── Node trace ────────────────────────────────────────────────────────────
    console.print(f"[dim]Node trace:[/dim] {' → '.join(node_trace)}")
    console.print(f"[dim]Reason:[/dim] {reason}")

    # ── Warnings ──────────────────────────────────────────────────────────────
    for w in warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")

    # ── Timing ───────────────────────────────────────────────────────────────
    if show_timing:
        timing = get_timing_report()
        t = Table(title="Model Timing", show_header=True, header_style="bold magenta")
        t.add_column("Metric")
        t.add_column("Value")
        for k, v in timing.items():
            t.add_row(str(k), str(v))
        console.print(t)

    console.print()


@click.group()
def cli():
    """OrbitDesk Support Agent CLI."""
    pass


@cli.command()
@click.argument("question")
@click.option("--json", "output_json", is_flag=True, help="Output raw JSON response")
@click.option("--timing", is_flag=True, help="Show model load and inference timing")
@click.option("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARNING)")
def ask(question: str, output_json: bool, timing: bool, log_level: str):
    """
    Ask the support agent a question.

    Examples:

        python -m src.cli ask "Our timezone changed and exports stopped. What do we check?"

        python -m src.cli ask "Can a Viewer create API credentials?" --json

        python -m src.cli ask "Our sync is broken" --timing
    """
    os.environ["LOG_LEVEL"] = log_level
    setup_logging()

    console.print(f"\n[bold]Question:[/bold] {question}\n")

    try:
        state = run_query(question)
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]Fatal error:[/bold red] {e}")
        sys.exit(1)

    if output_json:
        # Serialise state (drop non-serialisable objects)
        out = {
            "classification": state.get("classification"),
            "answer": state.get("final_answer"),
            "sources": state.get("sources", []),
            "confidence": state.get("confidence"),
            "requires_human": state.get("requires_human"),
            "reason": state.get("reason"),
            "clarifying_question": state.get("clarifying_question"),
            "escalation_reason": state.get("escalation_reason"),
            "warnings": state.get("warnings", []),
            "node_trace": state.get("node_trace", []),
            "attempt_count": state.get("attempt_count", 0),
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        format_response(state, show_timing=timing)


if __name__ == "__main__":
    cli()
