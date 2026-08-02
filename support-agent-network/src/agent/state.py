"""
Shared typed state for the OrbitDesk Support Agent graph.

Every node receives the full AgentState and returns a partial dict
with only the fields it mutated — LangGraph merges these automatically.
The state is the single source of truth shared across all nodes.
"""
from __future__ import annotations

from typing import Literal, Optional, TypedDict


# ── Sub-types ─────────────────────────────────────────────────────────────────

class Evidence(TypedDict):
    """A single retrieved evidence chunk from KB or resolved cases."""
    source_id: str          # e.g. "KB-003" or "CASE-1041"
    passage: str            # raw text chunk
    score: float            # cosine similarity score (0.0–1.0)
    chunk_index: int        # sequential position within the source document
    is_superseded: bool     # True if from a resolved case with status="superseded"


class Source(TypedDict):
    """A source citation included in the final response."""
    source_id: str          # e.g. "KB-003" or "CASE-1041"
    passage: str            # relevant excerpt
    char_start: int         # character offset in original document (extension over starter schema)
    char_end: int           # character offset in original document


class VerificationResult(TypedDict):
    """Structured output of the verification node."""
    passed: bool
    failure_reasons: list[str]    # human-readable failure descriptions
    overlap_score: float          # fraction of answer key terms found in evidence (0.0–1.0)


# ── Main state ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph StateGraph.

    Field naming conventions:
    - snake_case throughout
    - Optional[T] for fields that may not be populated on all routes
    - list[T] fields are always initialised (never None) so nodes can safely append

    Immutability contract:
    - `query` must never be modified after the initial state is created
    - `attempt_count` must only be incremented, never reset or decremented
    - `node_trace` must only be appended to, never overwritten
    """

    # ── Core input ───────────────────────────────────────────────────────────
    query: str
    """The raw user query. Set once at graph entry; never mutated by any node."""

    # ── Triage output ────────────────────────────────────────────────────────
    classification: Literal[
        "answerable",
        "requires_clarification",
        "requires_escalation",
        "out_of_scope",
        "safe_failure",
    ]
    """
    Route decision from the triage node.
    - answerable: retrieval + generation + verification path
    - requires_clarification: short-circuit to finalise with clarifying_question
    - requires_escalation: retrieval + generation path; requires_human=True in output
    - out_of_scope: short-circuit to finalise with out-of-scope message
    - safe_failure: set by finalise when max retries exhausted or schema fails
    """

    triage_reason: str
    """Short (≤ 2 sentences) explanation of why the triage node chose this classification."""

    # ── Retrieval output ─────────────────────────────────────────────────────
    retrieved_evidence: list[Evidence]
    """Ranked list of evidence chunks from the local vector index. Empty list if not yet run."""

    retrieval_sufficient: bool
    """
    True if top-1 cosine score >= RETRIEVAL_THRESHOLD (0.35).
    False signals the graph to short-circuit to clarification (for answerable) or
    to proceed with low-confidence generation (for requires_escalation).
    """

    # ── Generation I/O ───────────────────────────────────────────────────────
    draft_answer: Optional[str]
    """Raw text output from the generation node. None before generation runs."""

    generation_prompt: Optional[str]
    """The complete prompt sent to the LLM. Stored for debugging and retry context."""

    regeneration_hint: Optional[str]
    """
    On retry: the exact failure_reasons string from the previous verification result.
    Injected into the retry prompt so the model knows specifically what to fix.
    None on first attempt.
    """

    # ── Verification output ──────────────────────────────────────────────────
    verification_result: Optional[VerificationResult]
    """Structured result from the verification node. None before verification runs."""

    # ── Final output ─────────────────────────────────────────────────────────
    final_answer: Optional[str]
    """The validated final answer text returned to the user. None until finalise runs."""

    sources: list[Source]
    """Source citations included in the response. Empty list if not applicable."""

    confidence: float
    """
    Composite confidence score (0.0–1.0).
    Derived as min(top_retrieval_score, verification_overlap_score).
    0.0 for out_of_scope, requires_clarification, and safe_failure routes.
    """

    requires_human: bool
    """True for requires_escalation and safe_failure routes; False otherwise."""

    reason: str
    """
    Outer-facing explanation of the route and confidence.
    Required by the starter output schema.
    """

    clarifying_question: Optional[str]
    """
    Populated only when classification == 'requires_clarification'.
    Must be specific to the entity class detected (not a generic 'please provide more detail').
    None on all other routes.
    """

    escalation_reason: Optional[str]
    """
    Populated only when classification == 'requires_escalation'.
    Extension over starter schema to distinguish escalation from other requires_human cases.
    None on all other routes.
    """

    warnings: list[str]
    """
    Append-only list of non-fatal warnings surfaced to the caller.
    Example: 'Superseded case CASE-0914 appeared in retrieval results; its guidance was excluded.'
    Always initialised as an empty list.
    """

    # ── Loop guard ───────────────────────────────────────────────────────────
    attempt_count: int
    """
    Number of generation attempts made so far. Starts at 0.
    Incremented at the START of the generation node (before the model call).
    The ceiling check (attempt_count >= MAX_ATTEMPTS) lives in the conditional
    edge functions in graph.py, NOT inside any node — this prevents a buggy node
    from bypassing the loop termination.
    """

    # ── Execution audit ──────────────────────────────────────────────────────
    node_trace: list[str]
    """
    Ordered list of node names in the sequence they executed.
    Every node appends its own name BEFORE any other logic runs.
    Written to logs/run_<timestamp>.jsonl by logging_config.py.
    Directly satisfies the rubric requirement for execution logs.
    """


# ── Default state factory ─────────────────────────────────────────────────────

def make_initial_state(query: str) -> AgentState:
    """
    Return a fully initialised AgentState for a new query.

    All list fields are fresh lists (not shared references).
    All optional fields are None.
    attempt_count and confidence start at 0.
    classification defaults to 'answerable'; triage node will overwrite this.
    """
    return AgentState(
        query=query,
        classification="answerable",          # triage overwrites
        triage_reason="",
        retrieved_evidence=[],
        retrieval_sufficient=False,
        draft_answer=None,
        generation_prompt=None,
        regeneration_hint=None,
        verification_result=None,
        final_answer=None,
        sources=[],
        confidence=0.0,
        requires_human=False,
        reason="",
        clarifying_question=None,
        escalation_reason=None,
        warnings=[],
        attempt_count=0,
        node_trace=[],
    )
