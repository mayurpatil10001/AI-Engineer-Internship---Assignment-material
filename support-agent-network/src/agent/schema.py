"""
Output schema for the OrbitDesk Support Agent response.

Extends the starter output_schema.json with:
  - sources[].char_start / char_end  (exact evidence span offsets)
  - escalation_reason                (distinguishes escalation from other requires_human responses)
  - classification includes 'safe_failure' terminal state

All final responses are validated against SupportResponse before being returned.
A ValidationError in validate_response() must route to safe_failure, never raise.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SourceCitation(BaseModel):
    """A cited evidence source in the final response."""

    source_id: str = Field(
        description="Knowledge-base document ID (e.g. 'KB-003') or resolved-case ID (e.g. 'CASE-1041')"
    )
    passage: str = Field(min_length=1, description="Relevant excerpt from the source")
    char_start: int = Field(
        default=0,
        ge=0,
        description="Character offset of passage start in original document (0 if unknown)",
    )
    char_end: int = Field(
        default=0,
        ge=0,
        description="Character offset of passage end in original document (0 if unknown)",
    )

    @field_validator("source_id")
    @classmethod
    def source_id_format(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("source_id must not be empty")
        return v


class SupportResponse(BaseModel):
    """
    Validated final response returned by the support agent.

    Required fields match the starter output_schema.json.
    Extensions (char_start/char_end, escalation_reason) are Optional with sensible defaults
    so the schema remains backward-compatible.
    """

    classification: Literal[
        "answerable",
        "requires_clarification",
        "requires_escalation",
        "out_of_scope",
        "safe_failure",
    ]

    answer: str = Field(min_length=1, description="The answer text returned to the user")

    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Sources cited in the answer. Empty for out_of_scope and requires_clarification.",
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Composite confidence score (min of retrieval and verification overlap scores)",
    )

    requires_human: bool = Field(
        description="True for requires_escalation and safe_failure routes"
    )

    reason: str = Field(
        min_length=1,
        description="Brief explanation of the routing decision and confidence level",
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Specific clarifying question when classification == requires_clarification. "
            "Must name exactly what information is needed, not a generic 'provide more detail'."
        ),
    )

    escalation_reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable escalation rationale when classification == requires_escalation. "
            "Extension over starter schema."
        ),
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings, e.g. superseded case detected in retrieval results",
    )

    @model_validator(mode="after")
    def check_classification_consistency(self) -> "SupportResponse":
        """Enforce classification-specific field requirements."""

        if self.classification == "answerable" and not self.sources:
            raise ValueError(
                "classification='answerable' requires at least one source citation"
            )

        if self.classification == "requires_clarification":
            if not self.clarification_question:
                raise ValueError(
                    "classification='requires_clarification' requires a non-empty clarification_question"
                )

        if self.classification == "requires_escalation":
            if not self.requires_human:
                raise ValueError(
                    "classification='requires_escalation' must have requires_human=True"
                )

        if self.classification == "safe_failure":
            if not self.requires_human:
                raise ValueError(
                    "classification='safe_failure' must have requires_human=True"
                )

        return self


# ── Sentinel text ─────────────────────────────────────────────────────────────

SAFE_FAILURE_ANSWER = (
    "I was unable to produce a verified answer from the available documentation. "
    "Please contact the OrbitDesk support team directly for assistance with this query."
)

CANNOT_ANSWER_SENTINEL = (
    "I cannot answer this question from the available documentation."
)


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_response(data: dict) -> SupportResponse:
    """
    Validate a dict of response fields against SupportResponse.

    Raises pydantic.ValidationError on failure — the caller (finalise node)
    is responsible for catching this and substituting a safe_failure response.

    Args:
        data: Dict with keys matching SupportResponse fields.

    Returns:
        A validated SupportResponse instance.
    """
    return SupportResponse.model_validate(data)


def make_safe_failure_response(reason: str, warnings: Optional[list[str]] = None) -> SupportResponse:
    """
    Build a guaranteed-valid safe_failure SupportResponse.

    This is the terminal fallback — it must never raise.
    """
    return SupportResponse(
        classification="safe_failure",
        answer=SAFE_FAILURE_ANSWER,
        sources=[],
        confidence=0.0,
        requires_human=True,
        reason=reason or "Verification failed after maximum retry attempts.",
        clarification_question=None,
        escalation_reason=None,
        warnings=warnings or [],
    )
