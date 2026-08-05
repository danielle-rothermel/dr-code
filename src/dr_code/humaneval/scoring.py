"""HumanEval submission scoring: parse, execute in the sandbox, classify.

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
    StrictStr,
)

from dr_code.humaneval.batch_runner import evaluate_humaneval_code
from dr_code.humaneval.code_parsing import (
    CodeExtractionResult,
    extract_code_with_profile,
)
from dr_code.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    resolve_humaneval_scoring_profile,
)
from dr_code.humaneval.sandbox import (
    SandboxRunner,
    run_python_in_sandbox,
)
from dr_code.humaneval.task import (
    EvaluationHarnessError,
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalTask,
)
from dr_code.base import FrozenModel

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


def score_humaneval_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
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
        evaluation = evaluate_humaneval_code(
            task=task,
            candidate_code=extraction.extracted_code,
            timeout_seconds=scoring_profile.timeout_seconds,
            candidate_ast=extraction.parsed_candidate,
            run_in_sandbox=run_in_sandbox,
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
