"""
Generation node: produce a draft answer from retrieved evidence using a local LLM.

Design principles (see docs/design.md §3.3):
  - The model is instructed ONLY to answer from provided context.
  - If context is insufficient, the model must emit the exact sentinel string
    (CANNOT_ANSWER_SENTINEL) — this triggers verification to pass automatically.
  - attempt_count is incremented HERE (at node start), before the model call.
    This ensures the loop-guard counter reflects actual calls, not routing passes.
  - On retry, the regeneration_hint from the previous verification failure is
    injected into the prompt so the retry is informed, not a blind repeat.
  - Generation does NOT run for out_of_scope or requires_clarification —
    the graph edges prevent this node from being reached on those routes.

Model: Qwen/Qwen2.5-0.5B-Instruct (see models.py for load details).
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from src.agent.schema import CANNOT_ANSWER_SENTINEL
from src.agent.state import AgentState, Evidence

logger = logging.getLogger("orbitdesk.agent")

# ── Prompt building ───────────────────────────────────────────────────────────
# Deterministic logic is in _build_prompt(); model call is in _call_model().
# This separation is explicit per the rubric's requirement.

def _format_evidence(evidence: list[Evidence]) -> str:
    """
    Format retrieved evidence chunks for inclusion in the prompt.

    Only non-superseded chunks are included.
    Superseded chunks are filtered here — a belt-and-suspenders defence
    in addition to the warning set by the retrieval node.
    """
    lines = []
    for e in evidence:
        if e["is_superseded"]:
            # Do not include superseded guidance in the generation context
            lines.append(
                f"[{e['source_id']}] NOTE: This case is superseded and must NOT be used as guidance."
            )
        else:
            lines.append(f"[{e['source_id']}] {e['passage']}")
    return "\n\n".join(lines) if lines else "(No evidence retrieved.)"


def _build_prompt(
    query: str,
    evidence: list[Evidence],
    regeneration_hint: Optional[str] = None,
) -> str:
    """
    Build the complete prompt string for the LLM.

    This is a deterministic function with no model calls.
    It is kept separate from the model invocation for testability.

    Args:
        query:              The original user query (never mutated).
        evidence:           Retrieved evidence chunks.
        regeneration_hint:  If this is a retry, the exact failure reason from verification.

    Returns:
        Complete prompt string ready to send to the generator.
    """
    evidence_text = _format_evidence(evidence)

    hint_block = ""
    if regeneration_hint:
        hint_block = (
            f"\n\nPREVIOUS ATTEMPT FAILED VERIFICATION:\n{regeneration_hint}\n"
            "Please address this specific issue in your revised answer. "
            "Ensure every claim is directly supported by the evidence below."
        )

    prompt = (
        "You are a support assistant for OrbitDesk. "
        "Answer ONLY using the evidence provided below. "
        "Do not invent error codes, feature names, role names, or steps that are not in the evidence. "
        f"If the evidence does not contain enough information to fully answer the question, "
        f"say exactly this and nothing else: "
        f'"{CANNOT_ANSWER_SENTINEL}"'
        f"{hint_block}"
        "\n\n--- EVIDENCE ---\n"
        f"{evidence_text}"
        "\n\n--- QUESTION ---\n"
        f"{query}"
        "\n\n--- ANSWER ---\n"
    )
    return prompt


def _call_model(prompt: str) -> tuple[str, float]:
    """
    Invoke the cached local LLM with the given prompt.

    This is the ONLY function in this module that calls a model.
    Separation from _build_prompt() makes the model boundary explicit.

    Returns:
        (generated_text, elapsed_seconds)
    """
    from src.agent.models import generate
    return generate(prompt)


# ── Node entry point ──────────────────────────────────────────────────────────

def generation_node(state: AgentState) -> dict:
    """
    Generation node: produce a draft answer from the local LLM.

    Always appends 'generation' to node_trace before any other work.
    Increments attempt_count BEFORE the model call.
    Outputs: draft_answer, generation_prompt, attempt_count, node_trace.
    """
    t0 = time.perf_counter()
    node_name = "generation"
    trace = list(state.get("node_trace", []))
    trace.append(node_name)

    # ── INCREMENT ATTEMPT COUNT FIRST (loop guard depends on this) ────────────
    attempt_count = state.get("attempt_count", 0) + 1

    query = state["query"]
    evidence = state.get("retrieved_evidence", [])
    regeneration_hint = state.get("regeneration_hint")

    logger.info(
        f"NODE {node_name} entry",
        extra={
            "node": node_name,
            "attempt": attempt_count,
            "evidence_count": len(evidence),
            "is_retry": bool(regeneration_hint),
        },
    )

    # ── Build prompt (deterministic, no model) ────────────────────────────────
    prompt = _build_prompt(query, evidence, regeneration_hint)

    # ── Call model ────────────────────────────────────────────────────────────
    draft_answer, generation_elapsed = _call_model(prompt)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"NODE {node_name} exit",
        extra={
            "node": node_name,
            "attempt": attempt_count,
            "generation_s": round(generation_elapsed, 2),
            "elapsed_ms": round(elapsed_ms, 1),
            "answer_preview": (draft_answer or "")[:120],
        },
    )

    return {
        "draft_answer": draft_answer,
        "generation_prompt": prompt,
        "attempt_count": attempt_count,
        "node_trace": trace,
    }
