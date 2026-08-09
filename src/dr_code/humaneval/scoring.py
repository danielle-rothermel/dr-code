from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from dr_exec import Executor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
)

from dr_code.core.execution.executor import (
    CompletedPythonProcess,
    ExecutionKilledError,
    ExecutionOutputLimitError,
)
from dr_code.humaneval.acceptance import (
    CodeExtractionResult,
    extract_humaneval_code,
)
from dr_code.humaneval import runner
from dr_code.preprocessing import PreprocessingFailureCode
from dr_code.humaneval.profiles import (
    HUMANEVAL_SCORING_PROFILE_ID,
    HUMANEVAL_SCORING_PROFILE_VERSION,
    HumanEvalScoringProfile,
    resolve_humaneval_scoring_profile,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationHarnessError,
    EvaluationCaseStatus,
    EvaluationTaskResult,
    HumanEvalTask,
)
from dr_code.core.models import FrozenModel
from dr_code.metrics.engine.execution import (
    ExecutionCache,
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
    is_killed_outcome,
    is_output_limit_outcome,
    is_timeout_outcome,
    run_requests,
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


@dataclass(frozen=True, slots=True)
class HumanEvalSubmissionRequest:
    raw_submission: str
    task: HumanEvalTask
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION


@dataclass(frozen=True, slots=True)
class _PlannedSubmission:
    request: HumanEvalSubmissionRequest
    extraction: CodeExtractionResult
    scoring_profile: HumanEvalScoringProfile
    function_names: tuple[str, ...]
    execution_requests: tuple[ExecutionRequest, ...]
    total_cases: int


async def score_humaneval_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    scoring_profile_id: str = HUMANEVAL_SCORING_PROFILE_ID,
    scoring_profile_version: str = HUMANEVAL_SCORING_PROFILE_VERSION,
    executor: Executor | None = None,
    execution_cache: ExecutionCache | None = None,
) -> HumanEvalSubmissionScore:
    scores = await score_humaneval_submissions_batch(
        (
            HumanEvalSubmissionRequest(
                raw_submission=raw_submission,
                task=task,
                scoring_profile_id=scoring_profile_id,
                scoring_profile_version=scoring_profile_version,
            ),
        ),
        executor=executor,
        execution_cache=execution_cache,
    )
    return scores[0]


async def score_humaneval_submissions_batch(
    requests: Sequence[HumanEvalSubmissionRequest],
    *,
    executor: Executor | None = None,
    execution_cache: ExecutionCache | None = None,
) -> tuple[HumanEvalSubmissionScore, ...]:
    """Score submissions through one execution-planning batch."""
    prepared: list[HumanEvalSubmissionScore | _PlannedSubmission] = []
    execution_requests: list[ExecutionRequest] = []
    for request in requests:
        if not isinstance(request, HumanEvalSubmissionRequest):
            raise TypeError(
                "requests must contain HumanEvalSubmissionRequest values"
            )
        item = _prepare_submission(request)
        prepared.append(item)
        if isinstance(item, _PlannedSubmission):
            execution_requests.extend(item.execution_requests)

    cache = (
        execution_cache
        if execution_cache is not None
        else InMemoryExecutionCache()
    )
    try:
        outcomes = await run_requests(
            execution_requests,
            executor=executor,
            cache=cache,
        )
    except Exception as exc:
        return tuple(
            item
            if not isinstance(item, _PlannedSubmission)
            else (
                _score_planned_submission(item, {})
                if not item.execution_requests
                else _harness_failure_score(item, exc)
            )
            for item in prepared
        )

    return tuple(
        item
        if not isinstance(item, _PlannedSubmission)
        else _score_planned_submission(item, outcomes)
        for item in prepared
    )


def _prepare_submission(
    request: HumanEvalSubmissionRequest,
) -> HumanEvalSubmissionScore | _PlannedSubmission:
    raw_submission = request.raw_submission
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    scoring_profile = resolve_humaneval_scoring_profile(
        scoring_profile_id=request.scoring_profile_id,
        scoring_profile_version=request.scoring_profile_version,
    )
    extraction = extract_humaneval_code(raw_submission)
    if extraction.accepted_code is None:
        return CompletedScore(
            raw_submission=raw_submission,
            extraction=extraction,
            outcome=extraction_failure_outcome(extraction),
            score=scoring_profile.metrics_profile.failed_score,
            evaluation=None,
            scoring_profile=scoring_profile,
        )

    try:
        function_names = tuple(
            runner.top_level_function_names(
                extraction.accepted_code,
                parsed_module=extraction.accepted_tree,
            )
        )
        total_cases = len(runner.require_parsed_tests(request.task).cases)
        execution_requests = tuple(
            _execution_request(
                task=request.task,
                candidate_code=extraction.accepted_code,
                function_name=function_name,
                timeout_seconds=scoring_profile.timeout_seconds,
            )
            for function_name in function_names
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
    return _PlannedSubmission(
        request=request,
        extraction=extraction,
        scoring_profile=scoring_profile,
        function_names=function_names,
        execution_requests=execution_requests,
        total_cases=total_cases,
    )


def _execution_request(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
) -> ExecutionRequest:
    request = runner.build_humaneval_batch_request(
        task=task,
        candidate_code=candidate_code,
        function_name=function_name,
        timeout_seconds=timeout_seconds,
    )
    return ExecutionRequest(
        source=request.source,
        input_json=request.input_json,
        timeout_seconds=request.timeout_seconds,
        computation_id=runner.HUMANEVAL_RUNNER_COMPUTATION_ID,
    )


def _score_planned_submission(
    planned: _PlannedSubmission,
    outcomes: Mapping[ExecutionRequest, ExecutionOutcome],
) -> HumanEvalSubmissionScore:
    try:
        evaluation = _evaluation_from_outcomes(planned, outcomes)
    except EvaluationHarnessError as exc:
        return _harness_failure_score(planned, exc)
    except Exception as exc:
        return _harness_failure_score(planned, exc)

    outcome = evaluation_outcome(evaluation)
    return CompletedScore(
        raw_submission=planned.request.raw_submission,
        extraction=planned.extraction,
        outcome=outcome,
        score=(
            planned.scoring_profile.metrics_profile.passed_score
            if outcome is SubmissionOutcome.PASSED
            else planned.scoring_profile.metrics_profile.failed_score
        ),
        evaluation=evaluation,
        scoring_profile=planned.scoring_profile,
    )


def _evaluation_from_outcomes(
    planned: _PlannedSubmission,
    outcomes: Mapping[ExecutionRequest, ExecutionOutcome],
) -> EvaluationTaskResult:
    results = []
    for function_name, request in zip(
        planned.function_names,
        planned.execution_requests,
        strict=True,
    ):
        try:
            results.extend(
                _case_results_from_outcome(
                    task=planned.request.task,
                    function_name=function_name,
                    request=request,
                    outcome=outcomes[request],
                )
            )
        except EvaluationHarnessError as exc:
            evaluation = EvaluationTaskResult(
                task_id=planned.request.task.task_id,
                entry_point=planned.request.task.entry_point,
                function_names=list(planned.function_names),
                total_cases=planned.total_cases,
                results=[*results, *exc.case_results],
            )
            raise EvaluationHarnessError(
                str(exc),
                case_results=exc.case_results,
                evaluation=evaluation,
                cause=exc.cause or exc,
            ) from exc
    return EvaluationTaskResult(
        task_id=planned.request.task.task_id,
        entry_point=planned.request.task.entry_point,
        function_names=list(planned.function_names),
        total_cases=planned.total_cases,
        results=results,
    )


def _case_results_from_outcome(
    *,
    task: HumanEvalTask,
    function_name: str,
    request: ExecutionRequest,
    outcome: ExecutionOutcome,
) -> list[EvaluationCaseResult]:
    if is_timeout_outcome(outcome):
        return runner.timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=request.timeout_seconds,
        )
    if is_output_limit_outcome(outcome):
        return runner.error_results(
            task=task,
            function_name=function_name,
            message=f"{ExecutionOutputLimitError.__name__}: {outcome.stderr}",
        )
    if is_killed_outcome(outcome):
        return runner.error_results(
            task=task,
            function_name=function_name,
            message=f"{ExecutionKilledError.__name__}: {outcome.stderr}",
        )
    return runner.interpret_subprocess_batch_result(
        task=task,
        function_name=function_name,
        completed=CompletedPythonProcess(
            returncode=outcome.returncode,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
        ),
        elapsed_seconds=0.0,
    )


def _harness_failure_score(
    planned: _PlannedSubmission,
    exc: Exception,
) -> HarnessFailure:
    if isinstance(exc, EvaluationHarnessError):
        cause = harness_failure_cause(exc)
        evaluation = exc.evaluation
    else:
        cause = HarnessFailureCause(
            exception_type=type(exc).__name__,
            message=str(exc),
        )
        evaluation = None
    return HarnessFailure(
        raw_submission=planned.request.raw_submission,
        extraction=planned.extraction,
        evaluation=evaluation,
        cause=cause,
        failure_class=UNKNOWN_FAILURE_CLASS,
        scoring_profile=planned.scoring_profile,
    )


def extraction_failure_outcome(
    extraction: CodeExtractionResult,
) -> SubmissionOutcome:
    if extraction.failure_code == PreprocessingFailureCode.BLANK_INPUT:
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
