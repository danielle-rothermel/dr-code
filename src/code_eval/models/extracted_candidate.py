"""ExtractedCandidate — a Code Candidate before Repair and Validation."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import ExtractorName


class ExtractedCandidate(FrozenModel):
    """One Code Candidate produced by Extraction."""

    source: str
    extractor: ExtractorName
    extractor_path: tuple[str, ...]
    notes: str = ""
