"""Serve-facade public API (FastAPI app lives behind the [serve] extra)."""

from dr_code.serve.explain import (
    ALL_EXPLAIN_STAGES,
    CandidateExplanation,
    CandidateStatus,
    ExplainStage,
    ExtractionExplanation,
    SelectionExplanation,
    UnwrapExplanation,
    explain_extraction,
)

__all__ = [
    "ALL_EXPLAIN_STAGES",
    "CandidateExplanation",
    "CandidateStatus",
    "ExplainStage",
    "ExtractionExplanation",
    "SelectionExplanation",
    "UnwrapExplanation",
    "explain_extraction",
]
