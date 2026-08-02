"""
Clarification stub — delegates to the finalise node.

See docs/design.md §8 for the design decision rationale:
The clarification response is fully deterministic (fixed template + clarifying_question
set by triage). Having a separate graph node would add a routing edge with no functional
difference. All clarification formatting lives in finalise_node().

This file exists to keep the directory structure matching the proposed scaffold
and to provide an import hook for testing.
"""
from src.agent.nodes.finalise import finalise_node as clarification_node  # noqa: F401

__all__ = ["clarification_node"]
