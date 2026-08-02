"""
Finalise node: format and validate the final response from AgentState.

This node is reached from multiple paths:
  - After verification passes (answerable, requires_escalation)
  - Directly after triage for out_of_scope, requires_clarification
  - After retrieval is insufficient for answerable (promoted to clarification)
  - After max retries are exhausted (safe_failure)

Responsibilities:
  1. Build the answer text for the classification/route combination.
  2. Convert retrieved_evidence to Source citations with char offsets.
  3. Call validate_response() to check the full pydantic schema.
  4. On ValidationError: substitute safe_failure response (never raises).
  5. Write the final run summary log entry.

The clarification stub (clarification.py) delegates to this node — see design.md §8.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from src.agent.schema import (
    SAFE_FAILURE_ANSWER,
    SupportResponse,
    make_safe_failure_response,
    validate_response,
)
from src.agent.state import AgentState, Evidence, Source

logger = logging.getLogger("orbitdesk.agent")


# ── Answer builders (deterministic, no model) ─────────────────────────────────

def _build_out_of_scope_answer(triage_reason: str) -> str:
    return (
        "This request is outside the scope of OrbitDesk support. "
        "OrbitDesk support can explain product behaviour, provide documented troubleshooting steps, "
        "and recommend escalation. It cannot issue refunds, provide legal or financial advice, "
        "cancel subscriptions, or perform actions outside the product documentation. "
        f"({triage_reason})"
    )


def _build_clarification_answer(clarifying_question: str) -> str:
    return (
        "To help you effectively, we need a few more details. "
        f"{clarifying_question}"
    )


def _build_escalation_answer(draft_answer: Optional[str], triage_reason: str) -> str:
    if draft_answer and len(draft_answer) > 20:
        return draft_answer
    return (
        "Based on the documented troubleshooting steps, this issue requires escalation to the "
        "OrbitDesk engineering team. "
        f"{triage_reason} "
        "Please collect the required diagnostic information before escalating."
    )


def _evidence_to_sources(evidence: list[Evidence]) -> list[Source]:
    """Convert Evidence dicts to Source citation dicts, filtering superseded."""
    seen: set[str] = set()
    sources: list[Source] = []
    for e in evidence:
        if e["is_superseded"]:
            continue
        # Deduplicate by source_id
        if e["source_id"] in seen:
            continue
        seen.add(e["source_id"])
        sources.append(Source(
            source_id=e["source_id"],
            passage=e["passage"][:300],   # cap passage length for schema output
            char_start=0,                 # offsets are 0 when not tracked at this level
            char_end=len(e["passage"]),
        ))
    return sources


# ── Node entry point ──────────────────────────────────────────────────────────

def finalise_node(state: AgentState) -> dict:
    """
    Finalise node: build and validate the final response.

    Always appends 'finalise' to node_trace before any other work.
    Never raises — ValidationError is caught and replaced with safe_failure.

    Outputs: final_answer, sources, requires_human, reason, confidence,
             clarifying_question, escalation_reason, warnings, classification,
             node_trace.
    """
    t0 = time.perf_counter()
    node_name = "finalise"
    trace = list(state.get("node_trace", []))
    trace.append(node_name)

    classification = state.get("classification", "safe_failure")
    triage_reason = state.get("triage_reason", "")
    draft_answer = state.get("draft_answer")
    verification_result = state.get("verification_result")
    evidence = state.get("retrieved_evidence", [])
    warnings = list(state.get("warnings", []))
    attempt_count = state.get("attempt_count", 0)
    clarifying_question = state.get("clarifying_question")

    logger.info(
        f"NODE {node_name} entry",
        extra={"node": node_name, "classification": classification},
    )

    # Determine if this is a safe_failure.
    # Rules:
    #   - classification already set to safe_failure by a prior node → always safe_fail
    #   - verification failed AND there is no draft answer → safe_fail
    #   - verification failed BUT we have evidence + a draft answer → return answerable
    #     with low confidence and a warning (better than silent safe_failure)
    verification_passed = (
        verification_result is not None and verification_result.get("passed", False)
    )
    verification_failed = (
        verification_result is not None and not verification_result.get("passed", False)
    )
    has_draft = bool(draft_answer and len(draft_answer) > 20)
    has_evidence = bool(evidence)

    is_safe_failure = classification == "safe_failure" or (
        verification_failed and not has_draft
    )

    if is_safe_failure:
        classification = "safe_failure"
    elif verification_failed and has_draft and has_evidence:
        # Verification failed but we have an answer grounded in evidence.
        # Return it with low confidence + warning rather than safe_failing.
        warnings.append(
            f"Answer did not fully pass verification "
            f"(overlap_score={verification_result.get('overlap_score', 0):.2f}). "
            "Returned with reduced confidence."
        )

    # ── Build answer text ─────────────────────────────────────────────────────
    answer_text: str
    requires_human: bool
    reason: str
    escalation_reason_out: Optional[str] = None

    if classification == "out_of_scope":
        answer_text = _build_out_of_scope_answer(triage_reason)
        requires_human = False
        reason = triage_reason or "Request is outside OrbitDesk support scope (KB-010)."

    elif classification == "requires_clarification":
        cq = clarifying_question or "Please provide more details about your issue."
        answer_text = _build_clarification_answer(cq)
        requires_human = False
        reason = triage_reason or "Insufficient information to diagnose the issue."

    elif classification == "requires_escalation":
        answer_text = _build_escalation_answer(draft_answer, triage_reason)
        requires_human = True
        reason = triage_reason or "Issue meets the escalation criteria defined in KB-004 and KB-008."
        escalation_reason_out = triage_reason

    elif classification == "answerable":
        answer_text = draft_answer or "No answer was generated."
        requires_human = False
        reason = (
            f"Answer generated from retrieved evidence "
            f"(overlap score: {(verification_result or {}).get('overlap_score', 0):.0%})."
        )

    else:  # safe_failure
        answer_text = SAFE_FAILURE_ANSWER
        requires_human = True
        failure_reasons = (
            verification_result["failure_reasons"]
            if verification_result and verification_result.get("failure_reasons")
            else []
        )
        reason = (
            f"Verification failed after {attempt_count} attempt(s). "
            + (failure_reasons[0] if failure_reasons else "Unknown failure.")
        )

    # ── Build sources ─────────────────────────────────────────────────────────
    sources = _evidence_to_sources(evidence) if classification in ("answerable", "requires_escalation") else []

    # ── Compute final confidence ──────────────────────────────────────────────
    confidence = state.get("confidence", 0.0)
    if classification in ("out_of_scope", "requires_clarification", "safe_failure"):
        confidence = 0.0
    elif verification_failed and has_draft:
        # Cap at 0.25 when answer returned despite failed verification
        confidence = min(confidence, 0.25)

    # ── Validate schema ───────────────────────────────────────────────────────
    response_dict = {
        "classification": classification,
        "answer": answer_text,
        "sources": [
            {
                "source_id": s["source_id"],
                "passage": s["passage"],
                "char_start": s["char_start"],
                "char_end": s["char_end"],
            }
            for s in sources
        ],
        "confidence": confidence,
        "requires_human": requires_human,
        "reason": reason,
        "clarification_question": clarifying_question,
        "escalation_reason": escalation_reason_out,
        "warnings": warnings,
    }

    try:
        validated: SupportResponse = validate_response(response_dict)
    except Exception as e:  # noqa: BLE001
        logger.error(
            "Schema validation failed in finalise — substituting safe_failure",
            extra={"error": str(e), "classification": classification},
        )
        warnings.append(f"Schema validation error: {e}")
        safe = make_safe_failure_response(
            reason=f"Schema validation error: {e}",
            warnings=warnings,
        )
        validated = safe

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"NODE {node_name} exit",
        extra={
            "node": node_name,
            "final_classification": validated.classification,
            "requires_human": validated.requires_human,
            "source_count": len(validated.sources),
            "elapsed_ms": round(elapsed, 1),
        },
    )

    return {
        "classification": validated.classification,
        "final_answer": validated.answer,
        "sources": [
            Source(
                source_id=s.source_id,
                passage=s.passage,
                char_start=s.char_start,
                char_end=s.char_end,
            )
            for s in validated.sources
        ],
        "confidence": validated.confidence,
        "requires_human": validated.requires_human,
        "reason": validated.reason,
        "clarifying_question": validated.clarification_question,
        "escalation_reason": validated.escalation_reason,
        "warnings": validated.warnings,
        "node_trace": trace,
    }
