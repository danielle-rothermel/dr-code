from __future__ import annotations

import ast
from collections.abc import Sequence
from enum import StrEnum, UNIQUE, verify
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import EvaluationSampleIdentity
from dr_code.evaluation.records import (
    CandidateJobCompleted,
    CandidateJobTerminated,
    CandidateTerminationReason,
    ExecutorExecutionFailure,
    HarnessExecutionFailure,
    NoCandidatesSampleRecord,
    PreprocessingAbsentSampleRecord,
    SampleEvaluationRecord,
)
from dr_code.evaluation.references import EvidenceReference
from dr_code.humaneval.job import (
    CandidateNamespaceFailure,
    HumanEvalSuiteCompleted,
    HumanEvalSuiteHarnessFailure,
)
from dr_code.humaneval.profiles import HumanEvalScoringProfile
from dr_code.humaneval.task import EvaluationCaseStatus, EvaluationTaskResult
from dr_code.metrics import OperatorFailureRecord
from dr_code.preprocessing import PreprocessingFailureCode


@verify(UNIQUE)
class SubmissionOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
    EMPTY_SUBMISSION = "empty_submission"
    EXTRACTION_FAILED = "extraction_failed"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"
    TIMED_OUT = "timed_out"


class HumanEvalSubmissionRequest(FrozenModel):
    sample: EvaluationSampleIdentity
    scoring_profile: HumanEvalScoringProfile


class HarnessFailureCause(FrozenModel):
    exception_type: str
    message: str


class CompletedSubmissionResult(FrozenModel):
    kind: Literal["completed"] = "completed"
    sample: EvaluationSampleIdentity
    outcome: SubmissionOutcome
    score: float
    evaluation: EvaluationTaskResult | None
    scoring_profile: HumanEvalScoringProfile
    sample_record: EvidenceReference


class HarnessFailure(FrozenModel):
    kind: Literal["harness_failure"] = "harness_failure"
    sample: EvaluationSampleIdentity
    evaluation: EvaluationTaskResult | None
    cause: HarnessFailureCause
    failure_class: str
    scoring_profile: HumanEvalScoringProfile
    sample_record: EvidenceReference


HumanEvalSubmissionResult: TypeAlias = Annotated[
    CompletedSubmissionResult | HarnessFailure,
    Field(discriminator="kind"),
]

_ReferencedSampleRecord: TypeAlias = tuple[
    SampleEvaluationRecord, EvidenceReference
]


def project_humaneval_submission(
    record: SampleEvaluationRecord,
    request: HumanEvalSubmissionRequest,
    /,
    *,
    sample_record: EvidenceReference,
) -> HumanEvalSubmissionResult:
    return project_humaneval_submissions_batch(
        ((record, sample_record),),
        (request,),
    )[0]


def project_humaneval_submissions_batch(
    records: Sequence[_ReferencedSampleRecord],
    requests: Sequence[HumanEvalSubmissionRequest],
    /,
) -> tuple[HumanEvalSubmissionResult, ...]:
    """Derive ordered benchmark results from authoritative sample records."""

    by_sample: dict[EvaluationSampleIdentity, _ReferencedSampleRecord] = {}
    for record, reference in records:
        identity = record.sample.identity
        if identity in by_sample:
            raise ValueError("sample evaluation records must be unique")
        by_sample[identity] = (record, reference)

    results: list[HumanEvalSubmissionResult] = []
    for request in requests:
        try:
            record, reference = by_sample[request.sample]
        except KeyError as error:
            raise ValueError(
                f"no sample evaluation record for {request.sample.sample_id!r}"
            ) from error
        results.append(_project_record(record, request, reference))
    return tuple(results)


def score_humaneval_submission(
    record: SampleEvaluationRecord,
    request: HumanEvalSubmissionRequest,
    /,
    *,
    sample_record: EvidenceReference,
) -> HumanEvalSubmissionResult:
    return project_humaneval_submission(
        record,
        request,
        sample_record=sample_record,
    )


def score_humaneval_submissions_batch(
    records: Sequence[_ReferencedSampleRecord],
    requests: Sequence[HumanEvalSubmissionRequest],
    /,
) -> tuple[HumanEvalSubmissionResult, ...]:
    return project_humaneval_submissions_batch(records, requests)


def _project_record(
    record: SampleEvaluationRecord,
    request: HumanEvalSubmissionRequest,
    reference: EvidenceReference,
) -> HumanEvalSubmissionResult:
    if isinstance(record, PreprocessingAbsentSampleRecord):
        outcome = (
            SubmissionOutcome.EMPTY_SUBMISSION
            if record.absence.failure_code
            == PreprocessingFailureCode.BLANK_INPUT.value
            else SubmissionOutcome.EXTRACTION_FAILED
        )
        return _completed(request, reference, outcome)
    if isinstance(record, NoCandidatesSampleRecord):
        return _completed(
            request,
            reference,
            SubmissionOutcome.EXTRACTION_FAILED,
        )

    if not record.candidates or not record.executions:
        return _harness_failure(
            request,
            reference,
            failure_type="MissingCandidateExecution",
            message="evaluated sample record has no candidate execution",
        )

    candidate = record.candidates[0]
    execution = record.executions[0]
    if execution.candidate != candidate.identity:
        return _harness_failure(
            request,
            reference,
            failure_type="InvalidCandidateExecution",
            message="first candidate execution does not match candidate zero",
        )
    if (
        candidate.identity.preprocessing
        != request.scoring_profile.preprocessing_definition
    ):
        return _harness_failure(
            request,
            reference,
            failure_type="UnsupportedPreprocessingDefinition",
            message="sample record preprocessing does not match the scoring profile",
        )

    if not record.metrics:
        return _harness_failure(
            request,
            reference,
            failure_type="MissingMetricEvidence",
            message="evaluated sample record has no first-candidate metric evidence",
        )
    metrics_definition = record.metrics[0].identity.metrics_definition
    expected_questions = metrics_definition.questions
    question_count = len(expected_questions)
    first_candidate_metrics = record.metrics[:question_count]
    if (
        len(record.metrics) != len(record.candidates) * question_count
        or tuple(
            metric.identity.question for metric in first_candidate_metrics
        )
        != expected_questions
        or any(
            metric.identity.metrics_definition != metrics_definition
            for metric in first_candidate_metrics
        )
    ):
        return _harness_failure(
            request,
            reference,
            failure_type="InvalidMetricEvidence",
            message="first-candidate metric slice is not exact",
        )
    operator_failure = next(
        (
            metric
            for metric in first_candidate_metrics
            if isinstance(metric, OperatorFailureRecord)
        ),
        None,
    )
    if operator_failure is not None:
        return _harness_failure(
            request,
            reference,
            failure_type=operator_failure.failure.failure_type,
            message=operator_failure.failure.failure_message,
        )
    selected_metrics = tuple(
        metric
        for metric in first_candidate_metrics
        if metric.identity.question == request.scoring_profile.question
    )
    if len(selected_metrics) != 1:
        return _harness_failure(
            request,
            reference,
            failure_type="UnsupportedMetricQuestion",
            message=(
                "first-candidate metric evidence does not contain exactly the "
                "HumanEval scoring-profile question"
            ),
        )
    outcome = execution.outcome
    if isinstance(outcome, HarnessExecutionFailure | ExecutorExecutionFailure):
        return _harness_failure(
            request,
            reference,
            failure_type=outcome.failure_type,
            message=outcome.message,
        )
    if isinstance(outcome, CandidateJobTerminated):
        projected = (
            SubmissionOutcome.TIMED_OUT
            if outcome.reason is CandidateTerminationReason.WALL_TIME
            else SubmissionOutcome.TESTS_FAILED
        )
        return _completed(request, reference, projected)
    if not isinstance(outcome, CandidateJobCompleted):
        return _harness_failure(
            request,
            reference,
            failure_type=type(outcome).__name__,
            message="unsupported candidate execution outcome",
        )
    if isinstance(outcome.result.namespace, CandidateNamespaceFailure):
        projected = (
            SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
            if not _top_level_functions(candidate.source.source)
            else SubmissionOutcome.TESTS_FAILED
        )
        return _completed(request, reference, projected)
    matching_suites = tuple(
        suite
        for suite in outcome.result.suites
        if suite.question == request.scoring_profile.question
    )
    if len(matching_suites) != 1:
        return _harness_failure(
            request,
            reference,
            failure_type="MissingHumanEvalSuite",
            message=(
                "candidate result does not contain exactly the HumanEval "
                "scoring-profile suite"
            ),
        )
    suite = matching_suites[0]
    if isinstance(suite, HumanEvalSuiteHarnessFailure):
        return _harness_failure(
            request,
            reference,
            failure_type=suite.failure_type,
            message=suite.message,
        )
    assert isinstance(suite, HumanEvalSuiteCompleted)
    cases = tuple(case for group in suite.groups for case in group.cases)
    if not outcome.result.namespace.function_names:
        projected = SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
    elif any(case.status is EvaluationCaseStatus.TIMEOUT for case in cases):
        projected = SubmissionOutcome.TIMED_OUT
    elif cases and all(
        case.status is EvaluationCaseStatus.PASSED for case in cases
    ):
        projected = SubmissionOutcome.PASSED
    elif not cases:
        projected = SubmissionOutcome.EVALUATION_INCOMPLETE
    else:
        projected = SubmissionOutcome.TESTS_FAILED
    return _completed(request, reference, projected)


def _completed(
    request: HumanEvalSubmissionRequest,
    reference: EvidenceReference,
    outcome: SubmissionOutcome,
) -> CompletedSubmissionResult:
    metrics = request.scoring_profile.metrics_profile
    return CompletedSubmissionResult(
        sample=request.sample,
        outcome=outcome,
        score=(
            metrics.passed_score
            if outcome is SubmissionOutcome.PASSED
            else metrics.failed_score
        ),
        evaluation=None,
        scoring_profile=request.scoring_profile,
        sample_record=reference,
    )


def _harness_failure(
    request: HumanEvalSubmissionRequest,
    reference: EvidenceReference,
    *,
    failure_type: str,
    message: str,
) -> HarnessFailure:
    return HarnessFailure(
        sample=request.sample,
        evaluation=None,
        cause=HarnessFailureCause(
            exception_type=failure_type,
            message=message,
        ),
        failure_class=failure_type,
        scoring_profile=request.scoring_profile,
        sample_record=reference,
    )


def _top_level_functions(source: str) -> tuple[str, ...]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


__all__ = [
    "CompletedSubmissionResult",
    "HarnessFailure",
    "HarnessFailureCause",
    "HumanEvalSubmissionRequest",
    "HumanEvalSubmissionResult",
    "SubmissionOutcome",
    "project_humaneval_submission",
    "project_humaneval_submissions_batch",
    "score_humaneval_submission",
    "score_humaneval_submissions_batch",
]
