"""Pipeline support steps.

Capture lives in `LLMCodeValidator.__init__` (fingerprint) and the first
lines of `validate()` (raw_input). Extraction is in `code_eval.extraction`.
Candidate Recovery is in `code_eval.candidate_recovery`.
Step 6 (normalize) is in `normalize_step.py`.
"""

from __future__ import annotations

from code_eval.models.candidate import Candidate
from code_eval.models.extraction_step import ExtractionStep

# ---------------------------------------------------------------------------
# Extraction log backfill
# ---------------------------------------------------------------------------


def backfill_extraction_log(
    extraction_log: tuple[ExtractionStep, ...],
    valid_candidates: tuple[Candidate, ...],
) -> tuple[ExtractionStep, ...]:
    """Mark extractors that contributed at least one valid candidate."""
    if not valid_candidates:
        return extraction_log

    valid_extractors = {name for candidate in valid_candidates for name in candidate.extractor_path}
    updated: list[ExtractionStep] = []
    for step in extraction_log:
        yielded = step.extractor.value in valid_extractors
        if yielded != step.yielded_valid_candidate:
            step = step.model_copy(update={"yielded_valid_candidate": yielded})
        updated.append(step)
    return tuple(updated)
