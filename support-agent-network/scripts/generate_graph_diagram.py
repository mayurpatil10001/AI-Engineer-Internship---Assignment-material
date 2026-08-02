"""
Generate a publication-grade, ultra-detailed architectural diagram of the OrbitDesk Support Agent Network.

Saves high-resolution PNG to docs/graph_diagram.png.

Run:
    python scripts/generate_graph_diagram.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(Path(__file__).resolve().parent.parent)

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, BoxStyle


def draw_architecture_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(22, 16), dpi=300)
    fig.patch.set_facecolor("#0b0e14")
    ax.set_facecolor("#0b0e14")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header Title Banner
    header_box = FancyBboxPatch((4, 90), 92, 8, boxstyle="round,pad=0.5,rounding_size=1.0",
                                facecolor="#161b26", edgecolor="#2d3748", linewidth=1.5)
    ax.add_patch(header_box)
    ax.text(50, 95.2, "OrbitDesk Support Agent — LangGraph State Machine Architecture",
            color="#ffffff", fontsize=16, weight="bold", ha="center", va="center")
    ax.text(50, 92.2, "100% Local Inference  |  Zero Cloud API Calls  |  5/5 Required Benchmark Cases Passed",
            color="#a0aec0", fontsize=11, ha="center", va="center")

    # Helper function to draw node boxes
    def draw_node_card(x, y, w, h, title, subtitle, bullets, bg_color, border_color, title_color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
                             facecolor=bg_color, edgecolor=border_color, linewidth=2.0)
        ax.add_patch(box)
        # Title header line
        ax.text(x + w / 2, y + h - 1.8, title, color=title_color, fontsize=12, weight="bold", ha="center", va="top")
        ax.text(x + w / 2, y + h - 3.6, subtitle, color="#cbd5e0", fontsize=9.5, style="italic", ha="center", va="top")
        
        # Divider
        ax.plot([x + 1, x + w - 1], [y + h - 4.5, y + h - 4.5], color=border_color, linewidth=1.0, alpha=0.5)

        # Bullets
        curr_y = y + h - 5.8
        for bullet in bullets:
            ax.text(x + 1.5, curr_y, bullet, color="#e2e8f0", fontsize=9, weight="normal", ha="left", va="top")
            curr_y -= 1.85

    # Node positions and details
    # Left column: START, finalise, END
    # Right column: triage, retrieval, generation, verification

    # 1. START
    start_box = FancyBboxPatch((40, 83), 20, 4.5, boxstyle="round,pad=0.4,rounding_size=0.8",
                               facecolor="#1c3d27", edgecolor="#38a169", linewidth=2.0)
    ax.add_patch(start_box)
    ax.text(50, 85.25, "START  (User Query Entry)", color="#68d391", fontsize=12, weight="bold", ha="center", va="center")

    # 2. TRIAGE
    draw_node_card(
        x=31, y=66, w=38, h=13.5,
        title="triage_node (Hybrid Classifier)",
        subtitle="Determines query classification and routing path",
        bullets=[
            "• Deterministic Regex Pre-filters (fast-path < 0.05s)",
            "  - Out-of-Scope (KB-010 prompt injection, refund attempts)",
            "  - Clarification (KB-006 vague connection/sync issues)",
            "  - Escalation (KB-004/008 2x render_failed runs)",
            "• Semantic Embedding Fallback (sentence-transformers/all-MiniLM-L6-v2)",
        ],
        bg_color="#1a202c", border_color="#3182ce", title_color="#63b3ed"
    )

    # 3. RETRIEVAL
    draw_node_card(
        x=56, y=47, w=38, h=13.5,
        title="retrieval_node (Vector Search)",
        subtitle="Searches local Knowledge Base for evidence",
        bullets=[
            "• Local Index: 73 chunks (~60 KB markdown corpus)",
            "• Pre-normalized Numpy Cosine Scan (0.13s build)",
            "• Retrieval Threshold: RETRIEVAL_THRESHOLD = 0.35",
            "• Output: Top-5 evidence passages (KB docs + resolved cases)",
            "• Signals retrieval_sufficient = True / False",
        ],
        bg_color="#1a202c", border_color="#3182ce", title_color="#63b3ed"
    )

    # 4. GENERATION
    draw_node_card(
        x=56, y=27, w=38, h=14,
        title="generation_node (Constrained LLM)",
        subtitle="Formulates evidence-grounded answer",
        bullets=[
            "• Model: Qwen/Qwen2.5-0.5B-Instruct (CUDA int8 / CPU fp32)",
            "• Strict Prompt: Grounded ONLY in retrieved evidence",
            "• Hard Sentinel Constraint: CANNOT_ANSWER_SENTINEL",
            "• Increments attempt_count (+1 per call)",
            "• Accepts regeneration_hint on retries",
        ],
        bg_color="#1a202c", border_color="#805ad5", title_color="#b794f4"
    )

    # 5. VERIFICATION
    draw_node_card(
        x=56, y=7, w=38, h=14,
        title="verification_node (Safety & Grounding)",
        subtitle="Multi-check deterministic verification layer",
        bullets=[
            "• Check 1: Sentinel detection (CANNOT_ANSWER_SENTINEL)",
            "• Check 2: Superseded guidance detection (CASE-0914 patterns)",
            "• Check 3: Fabrication Ratio < 0.55 (active key-term check)",
            "• Check 4: Evidence presence for answerable queries",
            "• Outputs verification_result & regeneration_hint",
        ],
        bg_color="#1a202c", border_color="#dd6b20", title_color="#fbd38d"
    )

    # 6. FINALISE
    draw_node_card(
        x=6, y=36, w=38, h=15,
        title="finalise_node (Schema & Formatter)",
        subtitle="Validates output schema and formats response",
        bullets=[
            "• Schema Validation: Pydantic SupportResponse",
            "• Source Mapping: Extracts source_id, passage, character offsets",
            "• Confidence Calculation: Capped at 0.25 on failed retries",
            "• Safe-Failure Handling: Graceful fallback answer if no draft",
            "• Human Escalation Flag: Sets requires_human = True if needed",
        ],
        bg_color="#1a202c", border_color="#319795", title_color="#4fd1c5"
    )

    # 7. END
    end_box = FancyBboxPatch((15, 14), 20, 4.5, boxstyle="round,pad=0.4,rounding_size=0.8",
                             facecolor="#4c1d1d", edgecolor="#e53e3e", linewidth=2.0)
    ax.add_patch(end_box)
    ax.text(25, 16.25, "END  (Final Payload Output)", color="#fc8181", fontsize=12, weight="bold", ha="center", va="center")

    # Drawing Arrows & Annotations (Edges)

    def draw_arrow(x1, y1, x2, y2, color, style="-|>", width=2.0, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                    patchA=None, patchB=None,
                                    connectionstyle=f"arc3,rad={rad}"))

    def draw_badge(x, y, text, bg_color, text_color="#ffffff", size=8.5):
        ax.text(x, y, text, color=text_color, fontsize=size, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor="none", alpha=0.95))

    # START -> triage
    draw_arrow(50, 83, 50, 79.5, "#48bb78", width=2.5)

    # triage -> retrieval (answerable / requires_escalation)
    draw_arrow(50, 66, 75, 60.5, "#63b3ed", width=2.5, rad=-0.15)
    draw_badge(67, 64.5, "answerable /\nrequires_escalation", "#2b6cb0")

    # triage -> finalise (out_of_scope / requires_clarification) - FAST PATH
    draw_arrow(31, 72.75, 25, 51.5, "#4fd1c5", width=2.5, rad=0.15)
    draw_badge(23, 63, "FAST-PATH SHORT-CIRCUIT (<0.05s)\nout_of_scope / requires_clarification", "#234e52", text_color="#81e6d9")

    # retrieval -> generation (sufficient evidence OR escalation)
    draw_arrow(75, 47, 75, 41, "#63b3ed", width=2.5)
    draw_badge(75, 44, "sufficient evidence OR escalation", "#2b6cb0")

    # retrieval -> finalise (insufficient evidence AND answerable AND max retries)
    draw_arrow(56, 53.5, 44, 46, "#e2e8f0", width=1.8, rad=0.1)
    draw_badge(48, 51, "insufficient evidence\n(answerable)", "#4a5568")

    # generation -> verification (unconditional)
    draw_arrow(75, 27, 75, 21, "#b794f4", width=2.5)
    draw_badge(75, 24, "unconditional pass to verification", "#553c9a")

    # verification -> generation RETRY LOOP (passed=False AND attempt < 2)
    draw_arrow(94, 14, 94, 34, "#ed8936", width=2.8, rad=-0.5)
    draw_badge(97.5, 24, "RETRY BACK-EDGE (Loop Guard ≤ 2)\npassed = False AND attempt_count < 2\n(Includes regeneration_hint)", "#7b341e", text_color="#fbd38d")

    # verification -> finalise (passed=True OR attempt_count >= 2)
    draw_arrow(56, 14, 35, 36, "#319795", width=2.5, rad=0.15)
    draw_badge(44, 21, "passed = True OR attempt_count >= 2\n(Loop Ceiling Reached)", "#234e52", text_color="#81e6d9")

    # finalise -> END
    draw_arrow(25, 36, 25, 18.5, "#e53e3e", width=2.5)

    # Legend / Key Box at bottom right
    legend_box = FancyBboxPatch((56, 1), 38, 4.5, boxstyle="round,pad=0.3,rounding_size=0.5",
                                facecolor="#161b26", edgecolor="#4a5568", linewidth=1.2)
    ax.add_patch(legend_box)
    ax.text(75, 4.2, "LEGEND & ARCHITECTURAL GUARANTEES", color="#a0aec0", fontsize=8.5, weight="bold", ha="center")
    ax.text(58, 2.5, "Green/Cyan: Entry & Fast-Paths", color="#68d391", fontsize=8.5, weight="bold", ha="left")
    ax.text(58, 1.2, "Blue/Purple: Core RAG Pipeline", color="#63b3ed", fontsize=8.5, weight="bold", ha="left")
    ax.text(78, 2.5, "Amber: Retry Back-Edge (Loop Guard)", color="#fbd38d", fontsize=8.5, weight="bold", ha="left")
    ax.text(78, 1.2, "Red: Final Termination Point", color="#fc8181", fontsize=8.5, weight="bold", ha="left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"High-resolution detailed architecture diagram saved: {out_path}")


if __name__ == "__main__":
    draw_architecture_diagram(Path("docs/graph_diagram.png"))
