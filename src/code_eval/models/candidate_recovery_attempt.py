"""CandidateRecoveryAttempt - one repair/validation attempt row."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.validation_outcome import ValidationOutcome


class CandidateRecoveryAttempt(FrozenModel):
    """One candidate recovery attempt with trace-ready provenance."""

    attempt_id: str
    extracted_index: int
    attempt_index: int
    candidate_id: str
    canonical_candidate_id: str | None = None
    source_before: str
    source_after: str
    repairs_applied: tuple[str, ...]
    changed: bool
    deduped: bool
    validation: tuple[ValidationOutcome, ...]
    is_valid: bool
