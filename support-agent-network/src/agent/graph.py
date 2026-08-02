"""
LangGraph StateGraph assembly for the OrbitDesk Support Agent.

This module assembles the graph, defines all conditional edge functions,
and exposes the compiled app as `compiled_app`.

Graph topology (matches docs/design.md §4 routing table exactly):
  START → triage
  triage → finalise               (out_of_scope, requires_clarification)
  triage → retrieval              (answerable, requires_escalation)
  retrieval → finalise            (insufficient evidence + answerable, OR loop guard)
  retrieval → generation          (sufficient evidence, OR escalation path)
  generation → verification       (unconditional)
  verification → finalise         (passed=True, OR attempt_count >= MAX_ATTEMPTS)
  verification → generation       (passed=False AND attempt_count < MAX_ATTEMPTS)
  finalise → END

Loop guard: ALL conditional edges that could route back to generation check
  attempt_count >= MAX_ATTEMPTS FIRST, before any other condition.
  This ensures that even a bug in a node cannot produce an infinite loop.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from langgraph.graph import END, START, StateGraph  # type: ignore

from src.agent.nodes.finalise import finalise_node
from src.agent.nodes.generation import generation_node
from src.agent.nodes.retrieval import retrieval_node
from src.agent.nodes.triage import triage_node
from src.agent.nodes.verification import verification_node
from src.agent.state import AgentState

logger = logging.getLogger("orbitdesk.agent")

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ATTEMPTS: int = int(os.getenv("MAX_ATTEMPTS", "2"))
"""
Maximum number of generation attempts (initial + retries).
Default 2 = one initial attempt + one retry.
Enforced in EDGE FUNCTIONS, not inside nodes.
"""


# ── Conditional edge functions ────────────────────────────────────────────────
# Each function receives the current state and returns the name of the next node.
# Loop guard is checked FIRST in every function that could route to generation.

def route_after_triage(
    state: AgentState,
) -> Literal["finalise", "retrieval"]:
    """
    Route after triage based on classification.

    out_of_scope + requires_clarification → finalise (short-circuit, no retrieval/generation)
    answerable + requires_escalation → retrieval
    """
    c = state["classification"]
    if c in ("out_of_scope", "requires_clarification"):
        return "finalise"
    return "retrieval"


def route_after_retrieval(
    state: AgentState,
) -> Literal["finalise", "generation"]:
    """
    Route after retrieval.

    Loop guard: if attempt_count >= MAX_ATTEMPTS, go to finalise immediately.
    answerable + insufficient evidence → finalise (will be formatted as clarification/safe_failure).
    requires_escalation + insufficient evidence → generation (still need to generate escalation guidance).
    Sufficient evidence → generation.
    """
    # PRIMARY loop guard (secondary — belt and suspenders)
    if state["attempt_count"] >= MAX_ATTEMPTS:
        logger.warning(
            "Loop guard triggered in route_after_retrieval",
            extra={"attempt_count": state["attempt_count"], "max": MAX_ATTEMPTS},
        )
        return "finalise"

    if not state["retrieval_sufficient"] and state["classification"] == "answerable":
        # Insufficient evidence for a direct answer — route to finalise
        # finalise_node will detect the missing evidence and format accordingly
        return "finalise"

    return "generation"


def route_after_verification(
    state: AgentState,
) -> Literal["finalise", "generation"]:
    """
    Route after verification.

    Loop guard (PRIMARY): if attempt_count >= MAX_ATTEMPTS, always go to finalise.
    Verification passed → finalise.
    Verification failed + under limit → generation (retry with regeneration_hint).
    """
    # PRIMARY loop guard — checked FIRST, unconditionally
    if state["attempt_count"] >= MAX_ATTEMPTS:
        vr = state.get("verification_result")
        passed = vr is not None and vr.get("passed", False)
        log_extra = {
            "attempt_count": state["attempt_count"],
            "max": MAX_ATTEMPTS,
            "verification_passed": passed,
        }
        if passed:
            logger.info(
                "route_after_verification: max attempts reached (verification passed), routing to finalise",
                extra=log_extra,
            )
        else:
            logger.warning(
                "route_after_verification: max attempts reached (verification failed), routing to finalise",
                extra=log_extra,
            )
        return "finalise"

    vr = state.get("verification_result")
    if vr is None or vr["passed"]:
        return "finalise"

    # Verification failed and we're under the attempt ceiling — retry
    return "generation"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble the LangGraph StateGraph with all nodes and edges.

    Returns:
        An uncompiled StateGraph (call .compile() to get the runnable app).
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    graph.add_node("triage", triage_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generation", generation_node)
    graph.add_node("verification", verification_node)
    graph.add_node("finalise", finalise_node)

    # ── Add edges ─────────────────────────────────────────────────────────────

    # Entry
    graph.add_edge(START, "triage")

    # triage → conditional
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"finalise": "finalise", "retrieval": "retrieval"},
    )

    # retrieval → conditional
    graph.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {"finalise": "finalise", "generation": "generation"},
    )

    # generation → verification (unconditional)
    graph.add_edge("generation", "verification")

    # verification → conditional (the only back-edge; loop-guarded)
    graph.add_conditional_edges(
        "verification",
        route_after_verification,
        {"finalise": "finalise", "generation": "generation"},
    )

    # finalise → END (unconditional terminal)
    graph.add_edge("finalise", END)

    return graph


def compile_graph():
    """
    Compile the StateGraph into a runnable LangGraph app.

    Returns:
        CompiledStateGraph (has .invoke(), .stream() methods).
    """
    graph = build_graph()
    app = graph.compile()
    logger.info("Graph compiled", extra={"max_attempts": MAX_ATTEMPTS})
    return app


# Module-level compiled app (lazy — avoids model loading at import time)
_compiled_app = None


def get_compiled_app():
    """
    Return the cached compiled app, compiling on first call.

    This is the recommended entry point for all callers.
    """
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = compile_graph()
    return _compiled_app
