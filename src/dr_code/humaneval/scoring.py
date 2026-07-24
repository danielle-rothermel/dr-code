"""Pure HumanEval scoring primitives.

`SubmissionOutcome` is part of the score contract so consumers can persist
why a submission scored zero without parsing error text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal

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
    EMPTY_SUBMISSION_ERROR,
    CodeExtractionResult,
    extract_code_with_profile,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.execution.subprocess import (
    PythonSubprocessRunner,
    run_python_subprocess,
)
from dr_code.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    resolve_humaneval_scoring_profile,
)
from dr_code.humaneval.task import (
    EvaluationHarnessError,
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalTask,
)
from dr_code.models import FrozenModel
from dr_code.trace import OUTPUT_KEY, is_absent

UNKNOWN_FAILURE_CLASS = "unknown"


class SubmissionOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
    EMPTY_SUBMISSION = "empty_submission"
    EXTRACTION_FAILED = "extraction_failed"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"
    TIMED_OUT = "timed_out"


class CompletedScore(FrozenModel):
    kind: Literal["completed"] = "completed"
    raw_submission: str
    extraction: CodeExtractionResult
    outcome: SubmissionOutcome
    score: float
    evaluation: EvaluationTaskResult | None = None
    scoring_profile: HumanEvalScoringProfile


class HarnessFailureCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_type: StrictStr
    message: StrictStr


class HarnessFailure(FrozenModel):
    kind: Literal["harness_failure"] = "harness_failure"
    raw_submission: str
    extraction: CodeExtractionResult | None = None
    evaluation: EvaluationTaskResult | None = None
    cause: HarnessFailureCause
    failure_class: StrictStr
    scoring_profile: HumanEvalScoringProfile


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
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    run_in_subprocess: PythonSubprocessRunner = run_python_subprocess,
) -> HumanEvalSubmissionScore:
    """Score one submission under an exact registered scoring profile."""
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=scoring_profile_id,
        scoring_profile_version=scoring_profile_version,
    )

    extraction = extract_code_with_profile(
        raw_submission,
        profile=scoring_profile.parser_profile,
    )
    if extraction.extracted_code is None:
        outcome = extraction_failure_outcome(extraction)
        return CompletedScore(
            raw_submission=raw_submission,
            extraction=extraction,
            outcome=outcome,
            score=scoring_profile.metrics_profile.failed_score,
            evaluation=None,
            scoring_profile=scoring_profile,
        )

    try:
        evaluation = evaluate_human_eval_code(
            task=task,
            candidate_code=extraction.extracted_code,
            timeout_seconds=scoring_profile.timeout_seconds,
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
            scoring_profile=scoring_profile,
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
            scoring_profile=scoring_profile,
        )

    outcome = evaluation_outcome(evaluation)
    return CompletedScore(
        raw_submission=raw_submission,
        extraction=extraction,
        outcome=outcome,
        score=(
            scoring_profile.metrics_profile.passed_score
            if outcome is SubmissionOutcome.PASSED
            else scoring_profile.metrics_profile.failed_score
        ),
        evaluation=evaluation,
        scoring_profile=scoring_profile,
    )


#: Preprocessing failure codes that name a scoring outcome of their own.
#: The pipeline decides a submission carries no top-level function while
#: filtering candidates, so that verdict reaches scoring as a failure code
#: rather than as an evaluation result. Any other failure code means the
#: submission simply produced no usable candidate.
_OUTCOME_BY_FAILURE_CODE: Final[
    dict[PreprocessingFailureCode, SubmissionOutcome]
] = {
    PreprocessingFailureCode.DECODER_OUTPUT_BLANK: (
        SubmissionOutcome.EMPTY_SUBMISSION
    ),
    PreprocessingFailureCode.NO_TOP_LEVEL_FUNCTION_CANDIDATE: (
        SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
    ),
}


def extraction_failure_outcome(
    extraction: CodeExtractionResult,
) -> SubmissionOutcome:
    """Name why extraction produced no code, from the trace's failure code.

    The preprocessing trace records a stable failure code for the step that
    gave up; scoring reads that code rather than matching on error text.
    """
    if extraction.extraction_error == EMPTY_SUBMISSION_ERROR:
        return SubmissionOutcome.EMPTY_SUBMISSION
    output = extraction.trace.value(OUTPUT_KEY)
    if is_absent(output):
        try:
            code = PreprocessingFailureCode(output.failure_code)
        except ValueError:
            return SubmissionOutcome.EXTRACTION_FAILED
        return _OUTCOME_BY_FAILURE_CODE.get(
            code, SubmissionOutcome.EXTRACTION_FAILED
        )
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
