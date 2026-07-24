"""Pure HumanEval scoring primitives.

`SubmissionOutcome` is part of the score contract so consumers can persist
why a submission scored zero without parsing error text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_code.humaneval.batch_runner import evaluate_human_eval_code
from dr_code.humaneval.code_parsing import (
    CodeExtractionResult,
    CodeParserProfile,
)
from dr_code.execution.subprocess import (
    PythonSubprocessRunner,
    run_python_subprocess,
)
from dr_code.humaneval.task import (
    EvaluationHarnessError,
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalTask,
)
from dr_code.preprocessing import (
    resolve_preprocessing_definition,
    run_preprocessing,
)
from dr_code.trace import (
    Absent,
    CodeArtifact,
    TextArtifact,
    serialize_trace,
)

UNKNOWN_FAILURE_CLASS = "unknown"


class SubmissionOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
    EMPTY_SUBMISSION = "empty_submission"
    EXTRACTION_FAILED = "extraction_failed"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"
    TIMED_OUT = "timed_out"


class CompletedScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["completed"] = "completed"
    raw_submission: str
    extraction: CodeExtractionResult
    outcome: SubmissionOutcome
    score: float
    evaluation: EvaluationTaskResult | None = None


class HarnessFailureCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_type: StrictStr
    message: StrictStr


class HarnessFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["harness_failure"] = "harness_failure"
    raw_submission: str
    extraction: CodeExtractionResult | None = None
    evaluation: EvaluationTaskResult | None = None
    cause: HarnessFailureCause
    failure_class: StrictStr


HumanEvalSubmissionScore = Annotated[
    CompletedScore | HarnessFailure,
    Field(discriminator="kind"),
]


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


def score_humaneval_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    parser_profile: CodeParserProfile,
    timeout_seconds: float,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
) -> HumanEvalSubmissionScore:
    """Score one submission through its canonical preprocessing definition."""
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")

    definition = resolve_preprocessing_definition(
        definition_id=parser_profile.profile_id,
        version=parser_profile.version,
    )
    trace = run_preprocessing(
        definition.materialize(),
        TextArtifact(text=raw_submission),
    )
    output = trace.value("output")
    extraction = CodeExtractionResult(
        raw_submission=raw_submission,
        extracted_code=(
            output.source if isinstance(output, CodeArtifact) else None
        ),
        extraction_error=(
            _extraction_error(raw_submission, output)
            if isinstance(output, Absent)
            else None
        ),
        trace=serialize_trace(trace),
    )
    if extraction.extracted_code is None:
        outcome = extraction_failure_outcome(extraction)
        return CompletedScore(
            raw_submission=raw_submission,
            extraction=extraction,
            outcome=outcome,
            score=0.0,
            evaluation=None,
        )

    try:
        evaluation = evaluate_human_eval_code(
            task=task,
            candidate_code=extraction.extracted_code,
            timeout_seconds=timeout_seconds,
            candidate_ast=extraction.parsed_candidate,
            run_in_subprocess=run_in_subprocess,
        )
    except EvaluationHarnessError as exc:
        return HarnessFailure(
            raw_submission=raw_submission,
            extraction=extraction,
            evaluation=exc.evaluation,
            cause=harness_failure_cause(exc),
            failure_class=UNKNOWN_FAILURE_CLASS,
        )
    except Exception as exc:
        return HarnessFailure(
            raw_submission=raw_submission,
            extraction=extraction,
            evaluation=None,
            cause=HarnessFailureCause(
                exception_type=type(exc).__name__,
                message=str(exc),
            ),
            failure_class=UNKNOWN_FAILURE_CLASS,
        )

    outcome = evaluation_outcome(evaluation)
    return CompletedScore(
        raw_submission=raw_submission,
        extraction=extraction,
        outcome=outcome,
        score=1.0 if outcome is SubmissionOutcome.PASSED else 0.0,
        evaluation=evaluation,
    )


def _extraction_error(raw_submission: str, output: Absent) -> str:
    if not raw_submission.strip():
        return "empty raw submission"
    return f"{output.failed_step}: {output.cause}"


def extraction_failure_outcome(
    extraction: CodeExtractionResult,
) -> SubmissionOutcome:
    if extraction.extraction_error == "empty raw submission":
        return SubmissionOutcome.EMPTY_SUBMISSION
    return SubmissionOutcome.EXTRACTION_FAILED


def evaluation_outcome(
    evaluation: EvaluationTaskResult,
) -> SubmissionOutcome:
    if not evaluation.function_names:
        return SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
    if evaluation.passed:
        return SubmissionOutcome.PASSED
    if any(
        result.status is EvaluationCaseStatus.TIMEOUT
        for result in evaluation.results
    ):
        return SubmissionOutcome.TIMED_OUT
    if not evaluation.coverage_complete and not evaluation.failures:
        return SubmissionOutcome.EVALUATION_INCOMPLETE
    return SubmissionOutcome.TESTS_FAILED


def harness_failure_cause(exc: EvaluationHarnessError) -> HarnessFailureCause:
    cause = exc.cause
    if cause is None or cause is exc:
        return HarnessFailureCause(
            exception_type=type(exc).__name__,
            message=str(exc),
        )
    return HarnessFailureCause(
        exception_type=type(cause).__name__,
        message=str(cause),
    )


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
