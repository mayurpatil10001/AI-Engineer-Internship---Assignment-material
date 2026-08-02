"""
Generate the graph diagram (docs/graph_diagram.png).

Uses matplotlib + networkx to render a clean directed graph of the
LangGraph StateGraph topology. The layout exactly matches the routing
table in docs/design.md §4.

Run:
    python scripts/generate_graph_diagram.py

Output: docs/graph_diagram.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    nodes = [
        ("START", "start"),
        ("triage", "node"),
        ("retrieval", "node"),
        ("generation", "node"),
        ("verification", "node"),
        ("finalise", "node"),
        ("END", "end"),
    ]
    for name, kind in nodes:
        G.add_node(name, kind=kind)

    # Unconditional edges
    G.add_edge("START", "triage", label="")
    G.add_edge("generation", "verification", label="unconditional")
    G.add_edge("finalise", "END", label="")

    # Conditional from triage
    G.add_edge("triage", "finalise", label="out_of_scope /\nrequires_clarification")
    G.add_edge("triage", "retrieval", label="answerable /\nrequires_escalation")

    # Conditional from retrieval
    G.add_edge("retrieval", "finalise", label="insufficient evidence\n(answerable) OR loop guard")
    G.add_edge("retrieval", "generation", label="sufficient evidence\nOR escalation")

    # Conditional from verification (the back-edge)
    G.add_edge("verification", "finalise", label="passed=True\nOR at max attempts")
    G.add_edge("verification", "generation", label="passed=False\nAND under limit")

    return G


def draw_graph(G: nx.DiGraph, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Spread layout: main path on right column, finalise short-circuits on left
    pos = {
        "START":        (6.0, 10.0),
        "triage":       (6.0,  8.5),
        "finalise":     (2.5,  5.5),   # left column — receives short-circuit edges
        "retrieval":    (6.0,  7.0),
        "generation":   (6.0,  5.5),
        "verification": (6.0,  4.0),
        "END":          (2.5,  2.0),
    }

    # Node colours by kind
    kind_colors = {
        "start": "#4CAF50",
        "end":   "#F44336",
        "node":  "#1E88E5",
    }
    node_colors = [kind_colors[G.nodes[n]["kind"]] for n in G.nodes]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=3500,
        alpha=0.92,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_color="white",
        font_size=11,
        font_weight="bold",
    )

    # Separate straight edges from the back-edge (verification → generation)
    straight_edges = [(u, v) for u, v in G.edges() if not (u == "verification" and v == "generation")]
    back_edge = [("verification", "generation")]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=straight_edges,
        edge_color="#90CAF9",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=1.8,
        connectionstyle="arc3,rad=0.0",
        min_source_margin=35,
        min_target_margin=35,
    )
    # Back-edge with arc
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=back_edge,
        edge_color="#FFB74D",
        arrows=True,
        arrowsize=20,
        arrowstyle="-|>",
        width=2.0,
        connectionstyle="arc3,rad=-0.5",
        min_source_margin=35,
        min_target_margin=35,
    )

    # Edge labels — offset for readability
    edge_labels = nx.get_edge_attributes(G, "label")
    # Only label conditional edges
    conditional_labels = {
        k: v for k, v in edge_labels.items() if v and v != "unconditional"
    }
    nx.draw_networkx_edge_labels(
        G, pos, ax=ax,
        edge_labels=conditional_labels,
        font_size=7.5,
        font_color="#E0E0E0",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1d2e", edgecolor="none", alpha=0.8),
        label_pos=0.45,
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color="#4CAF50", label="START"),
        mpatches.Patch(color="#1E88E5", label="Processing node"),
        mpatches.Patch(color="#F44336", label="END"),
        mpatches.Patch(color="#FFB74D", label="Retry back-edge (loop-guarded)"),
        mpatches.Patch(color="#90CAF9", label="Forward edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", facecolor="#1a1d2e",
              labelcolor="white", fontsize=9, framealpha=0.9)

    ax.set_title(
        "OrbitDesk Support Agent — LangGraph StateGraph\n"
        "(MAX_ATTEMPTS=2: verification→generation back-edge is loop-guarded)",
        color="white", fontsize=13, pad=15, weight="bold",
    )
    ax.axis("off")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Graph diagram saved: {out_path}")


if __name__ == "__main__":
    G = build_graph()
    draw_graph(G, Path("docs/graph_diagram.png"))
