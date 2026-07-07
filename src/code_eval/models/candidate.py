"""Candidate model — one extracted program with its full provenance."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import ExtractorName


class Candidate(FrozenModel):
    """One extraction attempt with the full trail needed for attribution.

    `extractor_path` is the ordered list of preprocessing/extraction step
    names that produced `source`. `repairs_applied` is the ordered list of
    repair-step names that were needed to turn `source` into a parseable
    program.
    """

    candidate_id: str
    source: str
    extractor: ExtractorName
    extractor_path: tuple[str, ...]
    repairs_applied: tuple[str, ...] = ()
    #: One outcome per validator that ran.
    validation: tuple[ValidationOutcome, ...] = ()
    #: True iff every validator that ran passed.
    is_valid: bool = False
