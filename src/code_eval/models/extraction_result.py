"""ExtractionResult — all Extraction output for one Raw LLM Output."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.extracted_candidate import ExtractedCandidate
from code_eval.models.extraction_pass import ExtractionPass
from code_eval.models.extraction_step import ExtractionStep


class ExtractionResult(FrozenModel):
    """Code Candidates plus Trace-ready Extraction details."""

    normalized_output: str
    candidates: tuple[ExtractedCandidate, ...]
    passes: tuple[ExtractionPass, ...]
    extraction_log: tuple[ExtractionStep, ...]
