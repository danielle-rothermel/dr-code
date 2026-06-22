"""Project ValidationResult to ParseOutcome."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from code_eval.models.extraction_step import ExtractionStep
from code_eval.models.validation_result import ValidationResult

from dr_code.models.attempts import AttemptRecord
from dr_code.models.outcomes import CodeEvalProvenance, ParseOutcome
from dr_code.parsing.config import default_validator

if TYPE_CHECKING:
    from code_eval import LLMCodeValidator

_SKIP_NO_VALID = "no_valid_candidate"


def parse_attempt(
    record: AttemptRecord,
    *,
    validator: LLMCodeValidator | None = None,
) -> ParseOutcome:
    """Parse one AttemptRecord via code-eval EXTRACTION_CONFIG."""
    active = validator or default_validator()
    started = time.perf_counter()
    result = active.validate(record.raw_output, task_id=record.task_id)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return project_validation_result(
        record,
        result,
        latency_ms=latency_ms,
    )


def project_validation_result(
    record: AttemptRecord,
    result: ValidationResult,
    *,
    latency_ms: float | None = None,
) -> ParseOutcome:
    """Map a code-eval ValidationResult onto ParseOutcome."""
    parse_success = result.overall_success
    extracted_code = result.best_valid_source()
    code_eval_prov: CodeEvalProvenance | None = None
    skip_reason: str | None = None

    if parse_success:
        best = result.best_valid_candidate()
        code_eval_prov = CodeEvalProvenance(
            config_fingerprint=result.config_fingerprint,
            extractor_path=best.extractor_path if best is not None else None,
            repairs_applied=best.repairs_applied if best is not None else None,
            extraction_log_summary=_summarize_extraction_log(
                result.extraction_log
            ),
        )
    else:
        skip_reason = _SKIP_NO_VALID

    return ParseOutcome(
        sample_id=record.sample_id,
        run_id=record.run_id,
        task_id=record.task_id,
        parse_success=parse_success,
        extracted_code=extracted_code,
        candidate_count=len(result.candidates),
        valid_count=len(result.valid_candidates),
        code_eval=code_eval_prov,
        skip_reason=skip_reason,
        latency_ms=latency_ms,
    )


def _summarize_extraction_log(
    steps: tuple[ExtractionStep, ...],
) -> tuple[str, ...]:
    summary: list[str] = []
    for step in steps:
        label = (
            f"{step.extractor.value}:produced={step.candidates_produced}:"
            f"valid={step.yielded_valid_candidate}"
        )
        if step.notes:
            label = f"{label}:{step.notes}"
        summary.append(label)
    return tuple(summary)
