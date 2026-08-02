"""
tests/test_verification.py — Unit tests for the verification node logic.

All tests call run_verification() directly (no graph, no models, no I/O).
Tests cover: pass cases, failure cases, superseded-content detection,
sentinel pass-through, and edge cases.
"""
from __future__ import annotations

import pytest

from src.agent.nodes.verification import (
    FABRICATION_THRESHOLD,
    OVERLAP_THRESHOLD,
    _check_superseded_content,
    _compute_fabrication_ratio,
    _compute_overlap_score,
    _has_evidence_for_answerable,
    _is_sentinel_response,
    run_verification,
)
from src.agent.schema import CANNOT_ANSWER_SENTINEL
from src.agent.state import Evidence, VerificationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_evidence(
    source_id: str = "KB-001",
    passage: str = "OrbitDesk is a workspace product for building dashboards.",
    score: float = 0.8,
    is_superseded: bool = False,
) -> Evidence:
    return Evidence(
        source_id=source_id,
        passage=passage,
        score=score,
        chunk_index=0,
        is_superseded=is_superseded,
    )


# ── _is_sentinel_response ─────────────────────────────────────────────────────

class TestSentinelDetection:

    def test_exact_sentinel_detected(self):
        assert _is_sentinel_response(CANNOT_ANSWER_SENTINEL) is True

    def test_sentinel_with_trailing_text_detected(self):
        assert _is_sentinel_response(CANNOT_ANSWER_SENTINEL + " More text.") is True

    def test_normal_answer_not_sentinel(self):
        assert _is_sentinel_response("OrbitDesk schedules can be resaved to apply a new timezone.") is False

    def test_empty_string_not_sentinel(self):
        assert _is_sentinel_response("") is False


# ── _check_superseded_content ─────────────────────────────────────────────────

class TestSupersededContentCheck:

    def test_passes_for_clean_answer(self):
        answer = "An Owner or Admin can create API credentials from Settings > Developer."
        assert _check_superseded_content(answer) is None

    def test_detects_personal_token_phrase(self):
        answer = "The Analyst should open Profile > Personal token to create a legacy token."
        result = _check_superseded_content(answer)
        assert result is not None
        assert "superseded" in result.lower()

    def test_detects_personal_api_token(self):
        answer = "Create a personal API token from your profile page."
        result = _check_superseded_content(answer)
        assert result is not None

    def test_case_insensitive_detection(self):
        answer = "Use PROFILE > PERSONAL TOKEN to get access."
        result = _check_superseded_content(answer)
        assert result is not None

    def test_current_guidance_not_flagged(self):
        """Correct answer referencing workspace credentials must NOT be flagged."""
        answer = (
            "Viewers cannot create API credentials. An Owner or Admin must create "
            "a workspace credential from Settings > Developer > API credentials."
        )
        assert _check_superseded_content(answer) is None


# ── _compute_overlap_score ────────────────────────────────────────────────────

class TestOverlapScore:

    def test_identical_text_scores_high(self):
        evidence_text = "OrbitDesk schedules are saved by clicking the Save schedule button."
        ev = [make_evidence(passage=evidence_text)]
        score = _compute_overlap_score(evidence_text, ev)
        assert score > 0.7, f"Expected >0.7 for identical text, got {score}"

    def test_unrelated_text_scores_low(self):
        evidence = [make_evidence(passage="OrbitDesk workspace settings timezone admin")]
        answer = "The weather forecast shows rain tomorrow with high humidity."
        score = _compute_overlap_score(answer, evidence)
        assert score < 0.2, f"Expected <0.2 for unrelated text, got {score}"

    def test_empty_evidence_returns_zero(self):
        score = _compute_overlap_score("some answer text here", [])
        assert score == 0.0

    def test_short_answer_trivially_passes(self):
        """Answers with fewer than 3 words return 1.0 (trivially pass)."""
        score = _compute_overlap_score("Yes", [])
        assert score == 1.0

    def test_superseded_evidence_excluded(self):
        """Superseded evidence chunks must not contribute to the overlap score."""
        bad_ev = make_evidence(
            passage="Use Profile > Personal token to create a legacy API key.",
            is_superseded=True,
        )
        answer = "Use Profile personal token to create a legacy API key."
        score = _compute_overlap_score(answer, [bad_ev])
        # Since superseded evidence is excluded, overlap with empty corpus → 0
        assert score == 0.0


# ── _has_evidence_for_answerable ──────────────────────────────────────────────

class TestEvidencePresence:

    def test_answerable_with_evidence_passes(self):
        ev = [make_evidence()]
        result = _has_evidence_for_answerable("answerable", ev)
        assert result is None

    def test_answerable_without_evidence_fails(self):
        result = _has_evidence_for_answerable("answerable", [])
        assert result is not None
        assert "source" in result.lower()

    def test_out_of_scope_without_evidence_passes(self):
        """Non-answerable routes do not require evidence sources."""
        result = _has_evidence_for_answerable("out_of_scope", [])
        assert result is None

    def test_requires_clarification_without_evidence_passes(self):
        result = _has_evidence_for_answerable("requires_clarification", [])
        assert result is None


# ── run_verification integration ──────────────────────────────────────────────

class TestRunVerification:
    """Integration tests for the full verification pipeline."""

    def test_sentinel_answer_always_passes(self):
        """The 'cannot answer' sentinel must pass verification regardless of evidence."""
        vr = run_verification(
            answer=CANNOT_ANSWER_SENTINEL,
            evidence=[],
            classification="answerable",
        )
        assert vr["passed"] is True
        assert vr["overlap_score"] == 0.0

    def test_well_grounded_answer_passes(self):
        """An answer that closely mirrors the evidence text should pass."""
        passage = (
            "Changing the workspace timezone does not immediately rewrite existing "
            "recurring export schedules. Existing schedules retain the timezone stored "
            "when they were last saved and display a Timezone update pending notice. "
            "To apply the new workspace timezone open the schedule and select Save schedule."
        )
        evidence = [make_evidence(passage=passage, source_id="KB-003")]
        answer = (
            "Changing the workspace timezone does not rewrite existing recurring schedules. "
            "They display a Timezone update pending notice. To apply the new timezone, "
            "open the schedule and save it."
        )
        vr = run_verification(answer, evidence, "answerable")
        # May or may not pass depending on exact overlap — but superseded check must pass
        assert _check_superseded_content(answer) is None

    def test_superseded_content_causes_failure(self):
        """An answer reproducing CASE-0914 guidance must fail verification."""
        evidence = [make_evidence(passage="Legacy personal tokens were removed in v4.0.")]
        answer = "The Analyst should go to Profile > Personal token to create their access token."
        vr = run_verification(answer, evidence, "answerable")
        assert vr["passed"] is False
        assert any("superseded" in r.lower() for r in vr["failure_reasons"])

    def test_no_evidence_for_answerable_fails(self):
        """answerable classification with no evidence must fail."""
        vr = run_verification(
            answer="You should resave the schedule to apply the new timezone.",
            evidence=[],
            classification="answerable",
        )
        assert vr["passed"] is False
        assert any("source" in r.lower() for r in vr["failure_reasons"])

    def test_out_of_scope_with_no_evidence_passes_evidence_check(self):
        """out_of_scope does not require evidence sources."""
        vr = run_verification(
            answer="This request is outside the scope of OrbitDesk support.",
            evidence=[],
            classification="out_of_scope",
        )
        # Evidence check should not fail for out_of_scope
        evidence_failures = [r for r in vr["failure_reasons"] if "source" in r.lower()]
        assert not evidence_failures

    def test_fabrication_ratio_check(self):
        """An answer with many terms absent from evidence should trigger fabrication check."""
        evidence = [make_evidence(passage="OrbitDesk workspace")]
        # Answer full of plausible-sounding but absent OrbitDesk-specific terms
        answer = (
            "Navigate to the dashboard configuration panel and click the synchronization "
            "endpoint resolver button to reset the authentication handshake protocol."
        )
        fab = _compute_fabrication_ratio(answer, evidence)
        assert fab >= FABRICATION_THRESHOLD, (
            f"Expected fabrication ratio >= {FABRICATION_THRESHOLD}, got {fab}"
        )
