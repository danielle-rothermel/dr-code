"""Pure HumanEval scoring primitives.

`GeneratedCodeOutcome` is part of the primitive score contract so later
append-only score attempts can persist why a generation scored zero without
parsing error text. The current v0 experiment writers still persist their
legacy scoring columns only; wiring this outcome into durable score-attempt
records belongs to the schema/scoring-profile stage.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_code.humaneval.code_parsing import (
    CodeExtractionResult,
    CodeParserProfile,
    extract_code_with_profile,
)
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalTask,
    evaluate_human_eval_code,
)


class GeneratedCodeOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
    EMPTY_GENERATION = "empty_generation"
    EXTRACTION_FAILED = "extraction_failed"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"


class HumanEvalGenerationScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_generation: str
    extraction: CodeExtractionResult
    outcome: GeneratedCodeOutcome
    score: float
    evaluation: EvaluationTaskResult | None = None


class EvaluationAggregateMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_names: tuple[StrictStr, ...]
    total_cases: StrictInt
    result_count: StrictInt
    passed_count: StrictInt
    failed_count: StrictInt
    error_count: StrictInt
    timeout_count: StrictInt
    failure_count: StrictInt
    passed: StrictBool
    status_counts: dict[StrictStr, StrictInt]


def score_humaneval_generation(
    *,
    raw_generation: Any,
    task: HumanEvalTask,
    parser_profile: CodeParserProfile,
    timeout_seconds: float,
    recordable_text: Callable[[Any], str],
) -> HumanEvalGenerationScore:
    """Score one raw generation under a parser profile.

    ``recordable_text`` is the injected canonical-text renderer for the
    persisted ``raw_generation`` field (dr-code stays serialization- and
    storage-free; whetstone injects its recording-boundary function).
    """
    canonical_terminal = recordable_text(raw_generation)
    extraction = extract_code_with_profile(
        raw_generation,
        profile=parser_profile,
    )
    if extraction.extracted_code is None:
        outcome = extraction_failure_outcome(extraction)
        return HumanEvalGenerationScore(
            raw_generation=canonical_terminal,
            extraction=extraction,
            outcome=outcome,
            score=0.0,
            evaluation=None,
        )

    evaluation = evaluate_human_eval_code(
        task=task,
        candidate_code=extraction.extracted_code,
        timeout_seconds=timeout_seconds,
    )
    outcome = evaluation_outcome(evaluation)
    return HumanEvalGenerationScore(
        raw_generation=canonical_terminal,
        extraction=extraction,
        outcome=outcome,
        score=1.0 if outcome is GeneratedCodeOutcome.PASSED else 0.0,
        evaluation=evaluation,
    )


def extraction_failure_outcome(
    extraction: CodeExtractionResult,
) -> GeneratedCodeOutcome:
    if extraction.extraction_error == "empty raw generation":
        return GeneratedCodeOutcome.EMPTY_GENERATION
    return GeneratedCodeOutcome.EXTRACTION_FAILED


def evaluation_outcome(
    evaluation: EvaluationTaskResult,
) -> GeneratedCodeOutcome:
    if not evaluation.function_names:
        return GeneratedCodeOutcome.NO_TOP_LEVEL_FUNCTIONS
    if evaluation.passed:
        return GeneratedCodeOutcome.PASSED
    if not evaluation.coverage_complete and not evaluation.failures:
        return GeneratedCodeOutcome.EVALUATION_INCOMPLETE
    return GeneratedCodeOutcome.TESTS_FAILED


def evaluation_aggregate_metrics(
    evaluation: EvaluationTaskResult,
) -> EvaluationAggregateMetrics:
    status_counts = evaluation.status_counts
    return EvaluationAggregateMetrics(
        function_names=tuple(evaluation.function_names),
        total_cases=evaluation.total_cases,
        result_count=len(evaluation.results),
        passed_count=status_counts.get(EvaluationCaseStatus.PASSED.value, 0),
        failed_count=status_counts.get(EvaluationCaseStatus.FAILED.value, 0),
        error_count=status_counts.get(EvaluationCaseStatus.ERROR.value, 0),
        timeout_count=status_counts.get(EvaluationCaseStatus.TIMEOUT.value, 0),
        failure_count=len(evaluation.failures),
        passed=evaluation.passed,
        status_counts=status_counts,
    )
