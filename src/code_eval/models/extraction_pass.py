"""ExtractionPass — Trace-ready output for one Extraction heuristic."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.extracted_candidate import ExtractedCandidate
from code_eval.names import ExtractorName


class ExtractionPass(FrozenModel):
    """Raw and text-normalized Code Candidates for one extractor."""

    extractor: ExtractorName
    raw_candidates: tuple[ExtractedCandidate, ...]
    normalized_candidates: tuple[ExtractedCandidate, ...]
