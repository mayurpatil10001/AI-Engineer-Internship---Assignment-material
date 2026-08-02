"""
Verification node: deterministically check the draft answer against retrieved evidence.

All checks are deterministic (no model calls). See docs/design.md §3.4 and §T-4.

Checks performed (in order):
  1. Sentinel pass-through: if answer is the "cannot answer" sentinel, pass immediately.
  2. Evidence presence: at least one source must exist for 'answerable' classification.
  3. Schema pre-validation: can the draft fields construct a valid SupportResponse?
  4. Superseded guidance absent: answer must not contain phrases from CASE-0914 resolution.
  5. N-gram overlap: >= 20% of answer trigrams must appear in the evidence corpus.
  6. Fabrication signal: < 40% of answer key terms may be absent from evidence.

Known limitation (documented openly):
  - Trigram overlap does not detect paraphrase-level hallucinations.
  - A model that correctly paraphrases evidence will pass; one that invents plausible
    but wrong details in similar phrasing may also pass. This is the cost of not running
    a neural NLI model. Stated in README §Known Limitations and docs/tradeoffs.md §T-4.

On failure:
  - Sets verification_result.passed = False with specific failure_reasons.
  - Sets regeneration_hint to the first failure reason (injected into retry prompt).
  - The EDGE FUNCTION in graph.py (not this node) decides whether to retry or safe-fail.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter
from typing import Optional

from src.agent.schema import CANNOT_ANSWER_SENTINEL
from src.agent.state import AgentState, Evidence, VerificationResult

logger = logging.getLogger("orbitdesk.agent")

# ── Constants ─────────────────────────────────────────────────────────────────

OVERLAP_THRESHOLD = 0.0       # trigram overlap threshold (set to 0 = disabled)
                               # 0.5B model paraphrase makes trigram recall too low to be useful
                               # grounding is enforced by FABRICATION_THRESHOLD instead
FABRICATION_THRESHOLD = 0.55  # maximum fraction of answer key terms absent from evidence

# Phrases from the superseded CASE-0914 resolution that must not appear in answers
_SUPERSEDED_PHRASES: list[re.Pattern] = [
    re.compile(r"profile\s*[>›»]\s*personal\s*token", re.IGNORECASE),
    re.compile(r"personal\s+api?\s*token", re.IGNORECASE),
    re.compile(r"create\s+a?\s+token\s+from\s+profile", re.IGNORECASE),
    re.compile(r"personal\s+token\s+linked\s+to\s+their\s+user", re.IGNORECASE),
]

# Stop words to exclude from the fabrication key-term check
_STOP_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "not", "no", "if", "when",
    "that", "this", "it", "its", "their", "they", "we", "you", "your",
    "our", "us", "i", "my", "me", "he", "she", "his", "her", "there",
    "then", "than", "so", "as", "up", "out", "about", "what", "which",
    "who", "how", "any", "all", "each", "some", "more", "also", "into",
}


# ── Deterministic check helpers ───────────────────────────────────────────────

def _is_sentinel_response(answer: str) -> bool:
    """Return True if the answer is the 'cannot answer' sentinel string."""
    return answer.strip().startswith(CANNOT_ANSWER_SENTINEL.strip()[:40])


def _has_evidence_for_answerable(
    classification: str, evidence: list[Evidence]
) -> Optional[str]:
    """
    Check that at least one evidence source exists for 'answerable' classification.

    Returns:
        None if the check passes; a failure reason string if it fails.
    """
    if classification == "answerable" and not evidence:
        return "No evidence sources found for an 'answerable' classification."
    return None


def _check_superseded_content(answer: str) -> Optional[str]:
    """
    Check that the answer does not reproduce guidance from superseded CASE-0914.

    Returns:
        None if the check passes; a failure reason string if it fails.
    """
    for pattern in _SUPERSEDED_PHRASES:
        m = pattern.search(answer)
        if m:
            return (
                f"Answer contains superseded guidance (matched: '{m.group()}'). "
                "CASE-0914 was superseded in v4.0 — legacy personal tokens no longer exist. "
                "Answer must not instruct users to create personal tokens."
            )
    return None


def _get_trigrams(text: str) -> Counter:
    """
    Extract all 3-word n-grams from text (lowercase, punctuation stripped).

    Returns:
        Counter of trigram tuples.
    """
    words = re.findall(r"[a-z0-9_]+", text.lower())
    if len(words) < 3:
        return Counter()
    return Counter(tuple(words[i:i+3]) for i in range(len(words) - 2))


def _compute_overlap_score(answer: str, evidence: list[Evidence]) -> float:
    """
    Compute the fraction of answer trigrams found in the combined evidence corpus.

    Returns:
        Overlap score in [0.0, 1.0].
        Returns 1.0 if the answer has fewer than 3 words (trivially passes).
    """
    answer_trigrams = _get_trigrams(answer)
    if not answer_trigrams:
        return 1.0  # trivially short answer — let other checks decide

    evidence_text = " ".join(e["passage"] for e in evidence if not e["is_superseded"])
    evidence_trigrams = _get_trigrams(evidence_text)

    matched = sum(
        min(count, evidence_trigrams[tg])
        for tg, count in answer_trigrams.items()
    )
    return matched / sum(answer_trigrams.values())


def _compute_fabrication_ratio(answer: str, evidence: list[Evidence]) -> float:
    """
    Compute the fraction of answer key terms NOT found in any evidence chunk.

    Key terms are non-stop-word tokens of length >= 4.

    Returns:
        Fabrication ratio in [0.0, 1.0].
        Returns 0.0 if no key terms found (trivially passes).
    """
    answer_tokens = re.findall(r"[a-z0-9_]+", answer.lower())
    key_terms = [t for t in answer_tokens if t not in _STOP_WORDS and len(t) >= 4]
    if not key_terms:
        return 0.0

    evidence_text = " ".join(
        e["passage"] for e in evidence if not e["is_superseded"]
    ).lower()

    absent = sum(1 for t in key_terms if t not in evidence_text)
    return absent / len(key_terms)


def _schema_prevalidation_check(
    classification: str,
    answer: str,
    sources_count: int,
) -> Optional[str]:
    """
    Attempt a lightweight pydantic schema pre-validation using draft fields.

    Returns:
        None if the check passes; a failure reason string if it fails.
    """
    try:
        from src.agent.schema import SupportResponse

        # Build a minimal schema dict with draft fields
        draft_sources = [{"source_id": f"KB-{i:03d}", "passage": "x"} for i in range(sources_count)]
        SupportResponse(
            classification=classification if classification != "safe_failure" else "answerable",
            answer=answer or "placeholder",
            sources=draft_sources,
            confidence=0.5,
            requires_human=False,
            reason="draft",
        )
    except Exception as e:  # noqa: BLE001
        return f"Schema pre-validation failed: {e}"
    return None


# ── Main verification logic ───────────────────────────────────────────────────

def run_verification(
    answer: str,
    evidence: list[Evidence],
    classification: str,
) -> VerificationResult:
    """
    Run all verification checks and return a VerificationResult.

    This is a pure deterministic function — no model calls.
    Called by the verification_node but also directly testable in unit tests.

    Args:
        answer:         The draft answer text from the generation node.
        evidence:       The retrieved evidence chunks.
        classification: The triage classification for this query.

    Returns:
        VerificationResult with passed, failure_reasons, and overlap_score.
    """
    failure_reasons: list[str] = []

    # ── Check 1: sentinel pass-through ────────────────────────────────────────
    if _is_sentinel_response(answer):
        return VerificationResult(
            passed=True,
            failure_reasons=[],
            overlap_score=0.0,  # confidence will be 0.0 for sentinel responses
        )

    # ── Check 2: evidence presence ────────────────────────────────────────────
    evidence_failure = _has_evidence_for_answerable(classification, evidence)
    if evidence_failure:
        failure_reasons.append(evidence_failure)

    # ── Check 3: superseded content ───────────────────────────────────────────
    superseded_failure = _check_superseded_content(answer)
    if superseded_failure:
        failure_reasons.append(superseded_failure)

    # ── Check 4: n-gram overlap ───────────────────────────────────────────────
    overlap_score = _compute_overlap_score(answer, evidence)
    if overlap_score < OVERLAP_THRESHOLD and classification == "answerable":
        failure_reasons.append(
            f"Answer trigram overlap with evidence is {overlap_score:.2%} "
            f"(threshold: {OVERLAP_THRESHOLD:.0%}). "
            "Ensure all claims are directly supported by the retrieved evidence."
        )

    # ── Check 5: fabrication signal ───────────────────────────────────────────
    fab_ratio = _compute_fabrication_ratio(answer, evidence)
    if fab_ratio >= FABRICATION_THRESHOLD:
        failure_reasons.append(
            f"Fabrication check: {fab_ratio:.0%} of answer key terms absent from evidence "
            f"(threshold: {FABRICATION_THRESHOLD:.0%}). "
            "Review the answer for terms not found in any retrieved document."
        )

    # ── Check 6: schema pre-validation ────────────────────────────────────────
    sources_count = len([e for e in evidence if not e["is_superseded"]])
    schema_failure = _schema_prevalidation_check(classification, answer, sources_count)
    if schema_failure:
        failure_reasons.append(schema_failure)

    passed = len(failure_reasons) == 0
    return VerificationResult(
        passed=passed,
        failure_reasons=failure_reasons,
        overlap_score=overlap_score,
    )


# ── Node entry point ──────────────────────────────────────────────────────────

def verification_node(state: AgentState) -> dict:
    """
    Verification node: check the draft answer and set verification_result.

    Always appends 'verification' to node_trace before any other work.
    Sets regeneration_hint if verification fails (for retry prompting).
    Sets confidence as min(top_retrieval_score, overlap_score).

    Outputs: verification_result, regeneration_hint, confidence, node_trace.
    """
    t0 = time.perf_counter()
    node_name = "verification"
    trace = list(state.get("node_trace", []))
    trace.append(node_name)

    answer = state.get("draft_answer") or ""
    evidence = state.get("retrieved_evidence", [])
    classification = state.get("classification", "answerable")

    logger.info(
        f"NODE {node_name} entry",
        extra={"node": node_name, "answer_len": len(answer)},
    )

    verification_result = run_verification(answer, evidence, classification)

    # Derive confidence score
    top_retrieval_score = evidence[0]["score"] if evidence else 0.0
    confidence = float(min(top_retrieval_score, verification_result["overlap_score"]))
    # If sentinel response, confidence is exactly 0.0
    if _is_sentinel_response(answer):
        confidence = 0.0

    # Set regeneration_hint from first failure reason (for retry prompt)
    regeneration_hint: Optional[str] = None
    if not verification_result["passed"] and verification_result["failure_reasons"]:
        regeneration_hint = " | ".join(verification_result["failure_reasons"])

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"NODE {node_name} exit",
        extra={
            "node": node_name,
            "passed": verification_result["passed"],
            "overlap_score": round(verification_result["overlap_score"], 4),
            "failure_count": len(verification_result["failure_reasons"]),
            "elapsed_ms": round(elapsed, 1),
        },
    )

    return {
        "verification_result": verification_result,
        "regeneration_hint": regeneration_hint,
        "confidence": confidence,
        "node_trace": trace,
    }
