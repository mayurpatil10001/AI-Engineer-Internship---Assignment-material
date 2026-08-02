"""
Triage node: classify the incoming query into one of the four required routes.

Design: hybrid deterministic + embedding-similarity approach.
  - Deterministic pre-filters run FIRST (regex/keyword) for categories that are
    unambiguous from surface form: out_of_scope and vague-sync clarification.
    This avoids LLM exposure for prompt-injection attempts (KB-010) and wastes no
    latency on obviously-classifiable queries.
  - Embedding-similarity classification runs ONLY for queries that pass all
    deterministic filters, using cosine similarity against hand-written exemplar phrases.
    The embedding model is already loaded for retrieval, so no second model is needed.

See docs/design.md §3.1 and §T-1 for full justification.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import numpy as np

from src.agent.state import AgentState

logger = logging.getLogger("orbitdesk.agent")

# ── Constants ─────────────────────────────────────────────────────────────────

# Regex patterns for deterministic out_of_scope detection (KB-010)
# Each pattern is tested case-insensitively against the full query.
_OUT_OF_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brefund\b", re.IGNORECASE),
    re.compile(r"\blegal\s+advice\b", re.IGNORECASE),
    re.compile(r"\bcancel\s+(my\s+)?subscription\b", re.IGNORECASE),
    re.compile(r"\bignore\b.{0,40}\b(documentation|instructions|rules|context)\b", re.IGNORECASE),
    re.compile(r"\bprompt\s*injection\b", re.IGNORECASE),
    re.compile(r"\bdisregard\b.{0,30}\b(above|previous|instructions)\b", re.IGNORECASE),
    re.compile(r"\bissue\s+a\s+refund\b", re.IGNORECASE),
    re.compile(r"\bwrite\s+legal\b", re.IGNORECASE),
]

# Regex patterns for deterministic requires_clarification fast-path (KB-006)
# KB-006 explicitly states "sync is not working" is not specific enough to diagnose.
_CLARIFICATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bsync\s+(is\s+)?not\s+working\b", re.IGNORECASE),
    re.compile(r"\bdata\s+sync\s+(is\s+)?(broken|not\s+working|failing|stopped)\b", re.IGNORECASE),
    re.compile(r"\bsync\s+(broke|stopped|failed)\b", re.IGNORECASE),
]

# Clarifying question for sync issues — specific per KB-006 §Troubleshooting
_SYNC_CLARIFYING_QUESTION = (
    "Please provide the following details so we can diagnose the connection issue: "
    "(1) Workspace ID, "
    "(2) Connection name or ID, "
    "(3) Current connection state (Active / Error / Reauthorization required / Disabled), "
    "(4) Time of last successful refresh, "
    "(5) Latest error code if shown, "
    "(6) Whether both manual and scheduled refreshes are affected. "
    "Do not share database passwords, OAuth tokens, or API secrets."
)

# Regex patterns for requires_escalation fast-path (KB-004 / KB-008)
_ESCALATION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\btwo\s+(consecutive\s+)?render.failed\b", re.IGNORECASE),
    re.compile(r"\brender.failed\b.{0,60}\btwice\b", re.IGNORECASE),
    re.compile(r"\brender.failed\b.{0,60}\b(again|consecutive|two.runs?|second.run)\b", re.IGNORECASE),
    re.compile(r"\btwo\s+(consecutive\s+)?runs?.{0,30}\brender.failed\b", re.IGNORECASE),
    # Catch Q-004 pattern: 'Two export runs in a row failed with render_failed'
    re.compile(r"\btwo\b.{0,80}\brender.failed\b", re.IGNORECASE),
    re.compile(r"\brender.failed\b.{0,80}\btwo\b", re.IGNORECASE),
    # Catch 'two consecutive', 'two runs', 'two export runs' near 'render_failed'
    re.compile(r"\brender.failed\b.{0,80}\bin\s+a\s+row\b", re.IGNORECASE),
    re.compile(r"\bin\s+a\s+row\b.{0,80}\brender.failed\b", re.IGNORECASE),
]

# Exemplar phrases for embedding-similarity classification of ambiguous queries.
# One phrase per class — cosine similarity to query determines classification.
_CLASS_EXEMPLARS: dict[str, str] = {
    "answerable": (
        "How do I configure scheduled exports and manage workspace timezone settings?"
    ),
    "requires_clarification": (
        "I have a problem with my connection or data sync and I don't know what's wrong."
    ),
    "requires_escalation": (
        "We have a critical recurring error that needs engineering team escalation."
    ),
    "out_of_scope": (
        "I want a refund and legal advice about billing and subscriptions."
    ),
}

# Similarity gap threshold: if the top class score is this close to the second,
# default to answerable (conservative — let retrieval decide if there's enough evidence).
_AMBIGUITY_GAP = 0.05


# ── Deterministic helpers ─────────────────────────────────────────────────────

def _check_out_of_scope(query: str) -> Optional[tuple[str, str]]:
    """
    Run deterministic out-of-scope regex checks.

    Returns:
        (classification, reason) tuple if a pattern matches; None otherwise.
    """
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        m = pattern.search(query)
        if m:
            return (
                "out_of_scope",
                f"Query matches out-of-scope pattern '{pattern.pattern}' "
                f"(matched: '{m.group()}'). Per KB-010, this is outside the OrbitDesk support scope.",
            )
    return None


def _check_clarification_needed(query: str) -> Optional[tuple[str, str, str]]:
    """
    Run deterministic clarification fast-path checks.

    Returns:
        (classification, reason, clarifying_question) tuple if a pattern matches; None otherwise.
    """
    for pattern in _CLARIFICATION_PATTERNS:
        m = pattern.search(query)
        if m:
            return (
                "requires_clarification",
                (
                    f"Query matches vague-sync pattern '{pattern.pattern}'. "
                    "Per KB-006, the phrase is not specific enough to diagnose. "
                    "Specific details are required."
                ),
                _SYNC_CLARIFYING_QUESTION,
            )
    return None


def _check_escalation_needed(query: str) -> Optional[tuple[str, str]]:
    """
    Run deterministic escalation fast-path checks.

    Returns:
        (classification, reason) tuple if a pattern matches; None otherwise.
    """
    for pattern in _ESCALATION_PATTERNS:
        m = pattern.search(query)
        if m:
            return (
                "requires_escalation",
                (
                    f"Query indicates two consecutive render_failed events "
                    f"(matched: '{m.group()}'). "
                    "Per KB-004 and KB-008, this meets the escalation threshold."
                ),
            )
    return None


# ── Embedding-similarity classification ──────────────────────────────────────

def _classify_by_embedding(query: str) -> tuple[str, str, float]:
    """
    Classify a query by cosine similarity to class exemplar phrases.

    This runs ONLY when all deterministic checks pass.
    Uses the already-loaded embedding model (no extra model load).

    Returns:
        (classification, reason, top_score)
    """
    from src.agent.models import embed

    all_texts = [query] + list(_CLASS_EXEMPLARS.values())
    class_names = list(_CLASS_EXEMPLARS.keys())

    vectors = embed(all_texts)
    query_vec = vectors[0]
    exemplar_vecs = vectors[1:]

    # Cosine similarity
    q_norm = np.linalg.norm(query_vec)
    scores = []
    for ev in exemplar_vecs:
        e_norm = np.linalg.norm(ev)
        if q_norm < 1e-9 or e_norm < 1e-9:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(query_vec, ev) / (q_norm * e_norm)))

    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    best_class = class_names[best_idx]

    # Ambiguity check: if the gap to second place is small, default to 'answerable'
    sorted_scores = sorted(scores, reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
    if gap < _AMBIGUITY_GAP and best_class != "answerable":
        best_class = "answerable"
        reason = (
            f"Embedding similarity ambiguous (gap={gap:.3f} < {_AMBIGUITY_GAP}); "
            f"defaulting to 'answerable' to let retrieval determine evidence sufficiency."
        )
    else:
        reason = (
            f"Embedding similarity: best class='{best_class}' "
            f"score={best_score:.3f} gap={gap:.3f}."
        )

    return best_class, reason, best_score


# ── Node entry point ──────────────────────────────────────────────────────────

def triage_node(state: AgentState) -> dict:
    """
    Triage node: classify the query and set routing fields in state.

    Always appends 'triage' to node_trace before any other work.
    Outputs: classification, triage_reason, clarifying_question (or None), node_trace.

    This node does NOT call the generation model.
    """
    t0 = time.perf_counter()
    node_name = "triage"
    trace = list(state.get("node_trace", []))
    trace.append(node_name)

    query = state["query"]
    logger.info(f"NODE {node_name} entry", extra={"node": node_name, "query_preview": query[:100]})

    classification: str
    triage_reason: str
    clarifying_question: Optional[str] = None

    # ── Step 1: deterministic out-of-scope check ──────────────────────────────
    result = _check_out_of_scope(query)
    if result:
        classification, triage_reason = result
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"NODE {node_name} exit",
            extra={
                "node": node_name,
                "classification": classification,
                "method": "deterministic_out_of_scope",
                "elapsed_ms": round(elapsed, 1),
            },
        )
        return {
            "classification": classification,
            "triage_reason": triage_reason,
            "clarifying_question": None,
            "node_trace": trace,
        }

    # ── Step 2: deterministic clarification fast-path ─────────────────────────
    result2 = _check_clarification_needed(query)
    if result2:
        classification, triage_reason, clarifying_question = result2
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"NODE {node_name} exit",
            extra={
                "node": node_name,
                "classification": classification,
                "method": "deterministic_clarification",
                "elapsed_ms": round(elapsed, 1),
            },
        )
        return {
            "classification": classification,
            "triage_reason": triage_reason,
            "clarifying_question": clarifying_question,
            "node_trace": trace,
        }

    # ── Step 3: deterministic escalation fast-path ────────────────────────────
    esc_result = _check_escalation_needed(query)
    if esc_result:
        classification, triage_reason = esc_result
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"NODE {node_name} exit",
            extra={
                "node": node_name,
                "classification": classification,
                "method": "deterministic_escalation",
                "elapsed_ms": round(elapsed, 1),
            },
        )
        return {
            "classification": classification,
            "triage_reason": triage_reason,
            "clarifying_question": None,
            "node_trace": trace,
        }

    # ── Step 4: embedding-similarity classification ───────────────────────────
    classification, triage_reason, _ = _classify_by_embedding(query)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"NODE {node_name} exit",
        extra={
            "node": node_name,
            "classification": classification,
            "method": "embedding_similarity",
            "elapsed_ms": round(elapsed, 1),
        },
    )
    return {
        "classification": classification,
        "triage_reason": triage_reason,
        "clarifying_question": clarifying_question,
        "node_trace": trace,
    }
