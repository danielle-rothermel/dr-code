"""HumanEval scoring over declared preprocessing candidates."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)

from dr_code.humaneval.batch_runner import evaluate_human_eval_code
from dr_code.humaneval.sandbox import SandboxRunner, run_python_in_sandbox
from dr_code.humaneval.task import (
    EvaluationCaseStatus,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalTask,
)
from dr_code.preprocessing import (
    BoundPreprocessingRunner,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import (
    CodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    is_absent,
)

UNKNOWN_FAILURE_CLASS = "unknown"


class SubmissionOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"
    TIMED_OUT = "timed_out"
    PREPROCESSING_FAILED = "preprocessing_failed"
    NO_CANDIDATES = "no_candidates"
    HARNESS_FAILURE = "harness_failure"


class HarnessFailureCause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exception_type: StrictStr
    message: StrictStr


class CompletedCandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["completed"] = "completed"
    candidate_index: StrictInt
    candidate_id: StrictStr
    candidate_code: StrictStr
    outcome: SubmissionOutcome
    score: float
    evaluation: EvaluationTaskResult


class CandidateHarnessFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["harness_failure"] = "harness_failure"
    candidate_index: StrictInt
    candidate_id: StrictStr
    candidate_code: StrictStr
    evaluation: EvaluationTaskResult | None = None
    cause: HarnessFailureCause
    failure_class: StrictStr


HumanEvalCandidateScore = Annotated[
    CompletedCandidateScore | CandidateHarnessFailure,
    Field(discriminator="kind"),
]


class CompletedScore(BaseModel):
    """All candidate results for one raw submission.

    Preprocessing determines which source candidates are safe to execute;
    candidate and function success remain application-level scoring policy.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: Literal["completed"] = "completed"
    raw_submission: str
    preprocessing: Trace
    preprocessing_failure_code: StrictStr | None = None
    candidates: tuple[HumanEvalCandidateScore, ...]
    outcome: SubmissionOutcome
    score: float

    @model_validator(mode="after")
    def _reject_indeterminate_score(self) -> Self:
        if self.outcome is SubmissionOutcome.HARNESS_FAILURE:
            raise ValueError("harness failure cannot carry a score")
        return self


class HarnessFailure(BaseModel):
    """Candidate evidence for a submission whose score is indeterminate."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kind: Literal["harness_failure"] = "harness_failure"
    raw_submission: str
    preprocessing: Trace
    candidates: tuple[HumanEvalCandidateScore, ...]
    outcome: Literal[SubmissionOutcome.HARNESS_FAILURE] = (
        SubmissionOutcome.HARNESS_FAILURE
    )


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


DEFAULT_HUMANEVAL_PREPROCESSING_RUNNER = bind_preprocessing(
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
)


def score_humaneval_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    timeout_seconds: float,
    preprocessing_runner: BoundPreprocessingRunner = (
        DEFAULT_HUMANEVAL_PREPROCESSING_RUNNER
    ),
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
) -> HumanEvalSubmissionScore:
    """Preprocess and score every returned candidate in deterministic order."""
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")

    preprocessing = preprocessing_runner.run(TextArtifact(text=raw_submission))
    output = preprocessing.value("output")
    if is_absent(output):
        return CompletedScore(
            raw_submission=raw_submission,
            preprocessing=preprocessing,
            preprocessing_failure_code=output.failure_code,
            candidates=(),
            outcome=SubmissionOutcome.PREPROCESSING_FAILED,
            score=0.0,
        )
    if not isinstance(output, CodeCandidateSetArtifact):
        raise TypeError(
            "HumanEval preprocessing must output a CodeCandidateSetArtifact"
        )
    if not output.candidates:
        return CompletedScore(
            raw_submission=raw_submission,
            preprocessing=preprocessing,
            candidates=(),
            outcome=SubmissionOutcome.NO_CANDIDATES,
            score=0.0,
        )

    candidates_list: list[HumanEvalCandidateScore] = []
    for index, candidate_code in enumerate(output.candidates):
        candidate_id = output.lineage_at(index).candidate_id
        if candidate_id is None:
            raise TypeError(
                "HumanEval preprocessing candidates must carry candidate_id"
            )
        candidates_list.append(
            score_humaneval_candidate(
                candidate_index=index,
                candidate_id=candidate_id,
                candidate_code=candidate_code,
                task=task,
                timeout_seconds=timeout_seconds,
                run_in_sandbox=run_in_sandbox,
            )
        )
    candidates = tuple(candidates_list)
    outcome = submission_outcome(candidates)
    if outcome is SubmissionOutcome.HARNESS_FAILURE:
        return HarnessFailure(
            raw_submission=raw_submission,
            preprocessing=preprocessing,
            candidates=candidates,
        )
    return CompletedScore(
        raw_submission=raw_submission,
        preprocessing=preprocessing,
        candidates=candidates,
        outcome=outcome,
        score=1.0 if outcome is SubmissionOutcome.PASSED else 0.0,
    )


def score_humaneval_candidate(
    *,
    candidate_index: int,
    candidate_id: str,
    candidate_code: str,
    task: HumanEvalTask,
    timeout_seconds: float,
    run_in_sandbox: SandboxRunner,
) -> HumanEvalCandidateScore:
    try:
        evaluation = evaluate_human_eval_code(
            task=task,
            candidate_code=candidate_code,
            timeout_seconds=timeout_seconds,
            run_in_sandbox=run_in_sandbox,
        )
    except EvaluationHarnessError as exc:
        return CandidateHarnessFailure(
            candidate_index=candidate_index,
            candidate_id=candidate_id,
            candidate_code=candidate_code,
            evaluation=exc.evaluation,
            cause=harness_failure_cause(exc),
            failure_class=UNKNOWN_FAILURE_CLASS,
        )
    except Exception as exc:
        return CandidateHarnessFailure(
            candidate_index=candidate_index,
            candidate_id=candidate_id,
            candidate_code=candidate_code,
            cause=HarnessFailureCause(
                exception_type=type(exc).__name__,
                message=str(exc),
            ),
            failure_class=UNKNOWN_FAILURE_CLASS,
        )

    outcome = evaluation_outcome(evaluation)
    return CompletedCandidateScore(
        candidate_index=candidate_index,
        candidate_id=candidate_id,
        candidate_code=candidate_code,
        outcome=outcome,
        score=1.0 if outcome is SubmissionOutcome.PASSED else 0.0,
        evaluation=evaluation,
    )


def submission_outcome(
    candidates: tuple[HumanEvalCandidateScore, ...],
) -> SubmissionOutcome:
    if not candidates:
        return SubmissionOutcome.NO_CANDIDATES
    completed = [
        candidate
        for candidate in candidates
        if isinstance(candidate, CompletedCandidateScore)
    ]
    if any(
        candidate.outcome is SubmissionOutcome.PASSED
        for candidate in completed
    ):
        return SubmissionOutcome.PASSED
    if len(completed) != len(candidates):
        return SubmissionOutcome.HARNESS_FAILURE
    if any(
        candidate.outcome is SubmissionOutcome.TIMED_OUT
        for candidate in completed
    ):
        return SubmissionOutcome.TIMED_OUT
    if any(
        candidate.outcome is SubmissionOutcome.EVALUATION_INCOMPLETE
        for candidate in completed
    ):
        return SubmissionOutcome.EVALUATION_INCOMPLETE
    if all(
        candidate.outcome is SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
        for candidate in completed
    ):
        return SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
    return SubmissionOutcome.TESTS_FAILED


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


__all__ = [
    "CandidateHarnessFailure",
    "CompletedCandidateScore",
    "CompletedScore",
    "EvaluationAggregateMetrics",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalCandidateScore",
    "HumanEvalSubmissionScore",
    "SubmissionOutcome",
    "evaluation_aggregate_metrics",
    "evaluation_outcome",
    "score_humaneval_candidate",
    "score_humaneval_submission",
    "submission_outcome",
]
