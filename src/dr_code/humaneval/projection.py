from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, UNIQUE, verify
from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import (
    EvaluationSampleIdentity,
    MaterializedEvaluationCandidate,
)
from dr_code.evaluation.records import (
    CandidateExecutionRecord,
    CandidateJobCompleted,
    CandidateJobTerminated,
    CandidateTerminationReason,
    EvaluatedSampleRecord,
    ExecutorExecutionFailure,
    FailureClass,
    HarnessExecutionFailure,
    NoCandidatesSampleRecord,
    PreprocessingAbsentSampleRecord,
    SampleEvaluationRecord,
    failure_class_of,
)
from dr_code.evaluation.references import EvidenceReference
from dr_code.humaneval.job import (
    CandidateNamespaceFailure,
    HumanEvalSuiteCompleted,
    HumanEvalSuiteHarnessFailure,
)
from dr_code.humaneval.profiles import (
    CandidateReduction,
    HumanEvalScoringProfile,
)
from dr_code.humaneval.task import EvaluationCaseStatus, EvaluationTaskResult
from dr_code.metrics import (
    MetricQuestionCoordinate,
    MetricsDefinitionCoordinate,
    OperatorFailureRecord,
)
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
    failure_class: FailureClass
    scoring_profile: HumanEvalScoringProfile
    sample_record: EvidenceReference


HumanEvalSubmissionResult: TypeAlias = Annotated[
    CompletedSubmissionResult | HarnessFailure,
    Field(discriminator="kind"),
]

_ReferencedSampleRecord: TypeAlias = tuple[
    SampleEvaluationRecord, EvidenceReference
]


@dataclass(frozen=True, slots=True)
class _CandidatePairing:
    """One candidate paired with the execution record that measured it."""

    candidate: MaterializedEvaluationCandidate
    execution: CandidateExecutionRecord


@dataclass(frozen=True, slots=True)
class _CandidateInvalid:
    """The record's candidate/execution pairing cannot support scoring."""

    failure_type: str
    message: str


@dataclass(frozen=True, slots=True)
class _SliceValid:
    """The candidate's metric slice is exact and fully measured."""


@dataclass(frozen=True, slots=True)
class _SliceOperatorFailure:
    """The candidate's measurement is broken, so its outcome is unknown."""

    failure_type: str
    message: str


@dataclass(frozen=True, slots=True)
class _SliceInvalid:
    """The candidate's metric slice does not match the scoring profile."""

    failure_type: str
    message: str


@dataclass(frozen=True, slots=True)
class _CandidateOutcome:
    """A candidate's own benchmark outcome, before any reduction."""

    outcome: SubmissionOutcome


@dataclass(frozen=True, slots=True)
class _CandidateHarnessFailure:
    """A candidate's measurement failed rather than producing an outcome."""

    failure_type: str
    message: str
    failure_class: FailureClass = FailureClass.HARNESS


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

    pairing = _candidate_pairing(record, request, ordinal=0)
    if isinstance(pairing, _CandidateInvalid):
        return _harness_failure(
            request,
            reference,
            failure_type=pairing.failure_type,
            message=pairing.message,
        )
    candidate = pairing.candidate
    execution = pairing.execution

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
    if len(record.metrics) != len(record.candidates) * question_count:
        return _harness_failure(
            request,
            reference,
            failure_type="InvalidMetricEvidence",
            message="first-candidate metric slice is not exact",
        )

    if (
        request.scoring_profile.candidate_reduction
        is CandidateReduction.ANY_CANDIDATE_PASSES
    ):
        return _project_any_candidate(
            record,
            request,
            reference,
            metrics_definition=metrics_definition,
            expected_questions=expected_questions,
            question_count=question_count,
        )

    metric_slice = _validated_metric_slice(
        record,
        request,
        ordinal=0,
        metrics_definition=metrics_definition,
        expected_questions=expected_questions,
        question_count=question_count,
    )
    if isinstance(metric_slice, _SliceInvalid):
        return _harness_failure(
            request,
            reference,
            failure_type=metric_slice.failure_type,
            message=metric_slice.message,
        )
    if isinstance(metric_slice, _SliceOperatorFailure):
        return _harness_failure(
            request,
            reference,
            failure_type=metric_slice.failure_type,
            message=metric_slice.message,
            failure_class=_slice_failure_class(execution),
        )
    classified = _classify_candidate(candidate, execution, request)
    if isinstance(classified, _CandidateHarnessFailure):
        return _harness_failure(
            request,
            reference,
            failure_type=classified.failure_type,
            message=classified.message,
            failure_class=classified.failure_class,
        )
    return _completed(request, reference, classified.outcome)


def _project_any_candidate(
    record: EvaluatedSampleRecord,
    request: HumanEvalSubmissionRequest,
    reference: EvidenceReference,
    *,
    metrics_definition: MetricsDefinitionCoordinate,
    expected_questions: tuple[MetricQuestionCoordinate, ...],
    question_count: int,
) -> HumanEvalSubmissionResult:
    """Score a pass when any candidate's any function group passes the suite.

    A candidate is one extracted representation, which carries every top-level
    function written alongside the solution, so the within-candidate rule is
    existential too: one function group passing the complete test-case set is a
    pass, and helpers that fail beside a correct solution do not mask it.

    Failure attribution survives both quantifiers: a candidate whose measurement
    is broken cannot be scored as a clean fail, because it might have passed.
    """

    first_broken: _CandidateHarnessFailure | None = None
    first_outcome: SubmissionOutcome | None = None
    for ordinal in range(len(record.candidates)):
        pairing = _candidate_pairing(record, request, ordinal=ordinal)
        if isinstance(pairing, _CandidateInvalid):
            return _harness_failure(
                request,
                reference,
                failure_type=pairing.failure_type,
                message=pairing.message,
            )
        candidate = pairing.candidate
        execution = pairing.execution

        metric_slice = _validated_metric_slice(
            record,
            request,
            ordinal=ordinal,
            metrics_definition=metrics_definition,
            expected_questions=expected_questions,
            question_count=question_count,
        )
        if isinstance(metric_slice, _SliceInvalid):
            return _harness_failure(
                request,
                reference,
                failure_type=metric_slice.failure_type,
                message=metric_slice.message,
            )
        if isinstance(metric_slice, _SliceOperatorFailure):
            if first_broken is None:
                first_broken = _CandidateHarnessFailure(
                    failure_type=metric_slice.failure_type,
                    message=metric_slice.message,
                    failure_class=_slice_failure_class(execution),
                )
            continue

        classified = _classify_candidate(
            candidate,
            execution,
            request,
            any_function_group=True,
        )
        if isinstance(classified, _CandidateHarnessFailure):
            if first_broken is None:
                first_broken = classified
            continue
        if classified.outcome is SubmissionOutcome.PASSED:
            return _completed(request, reference, SubmissionOutcome.PASSED)
        if first_outcome is None:
            first_outcome = classified.outcome

    if first_broken is not None:
        return _harness_failure(
            request,
            reference,
            failure_type=first_broken.failure_type,
            message=first_broken.message,
            failure_class=first_broken.failure_class,
        )
    assert first_outcome is not None
    return _completed(request, reference, first_outcome)


def _candidate_pairing(
    record: EvaluatedSampleRecord,
    request: HumanEvalSubmissionRequest,
    *,
    ordinal: int,
) -> _CandidatePairing | _CandidateInvalid:
    if ordinal >= len(record.executions):
        return _CandidateInvalid(
            failure_type="InvalidCandidateExecution",
            message=(
                f"candidate {ordinal} has no matching candidate execution"
            ),
        )
    candidate = record.candidates[ordinal]
    execution = record.executions[ordinal]
    if execution.candidate != candidate.identity:
        return _CandidateInvalid(
            failure_type="InvalidCandidateExecution",
            message=(
                f"candidate execution {ordinal} does not match candidate "
                f"{ordinal}"
            ),
        )
    if (
        candidate.identity.preprocessing
        != request.scoring_profile.preprocessing_definition
    ):
        return _CandidateInvalid(
            failure_type="UnsupportedPreprocessingDefinition",
            message=(
                "sample record preprocessing does not match the scoring profile"
            ),
        )
    return _CandidatePairing(candidate=candidate, execution=execution)


def _validated_metric_slice(
    record: EvaluatedSampleRecord,
    request: HumanEvalSubmissionRequest,
    *,
    ordinal: int,
    metrics_definition: MetricsDefinitionCoordinate,
    expected_questions: tuple[MetricQuestionCoordinate, ...],
    question_count: int,
) -> _SliceValid | _SliceOperatorFailure | _SliceInvalid:
    start = ordinal * question_count
    candidate_metrics = record.metrics[start : start + question_count]
    if tuple(
        metric.identity.question for metric in candidate_metrics
    ) != expected_questions or any(
        metric.identity.metrics_definition != metrics_definition
        for metric in candidate_metrics
    ):
        return _SliceInvalid(
            failure_type="InvalidMetricEvidence",
            message=f"candidate {ordinal} metric slice is not exact",
        )
    operator_failure = next(
        (
            metric
            for metric in candidate_metrics
            if isinstance(metric, OperatorFailureRecord)
        ),
        None,
    )
    if operator_failure is not None:
        return _SliceOperatorFailure(
            failure_type=operator_failure.failure.failure_type,
            message=operator_failure.failure.failure_message,
        )
    selected_metrics = tuple(
        metric
        for metric in candidate_metrics
        if metric.identity.question == request.scoring_profile.question
    )
    if len(selected_metrics) != 1:
        return _SliceInvalid(
            failure_type="UnsupportedMetricQuestion",
            message=(
                f"candidate {ordinal} metric evidence does not contain exactly "
                "the HumanEval scoring-profile question"
            ),
        )
    return _SliceValid()


def _slice_failure_class(execution: CandidateExecutionRecord) -> FailureClass:
    """Attribute a broken metric slice to the party that owns the execution.

    The operator that measures a candidate fails whenever the execution itself
    failed, so the slice's failure carries the execution outcome's attribution
    rather than the measuring harness's.
    """

    return failure_class_of(execution.outcome) or FailureClass.HARNESS


def _classify_candidate(
    candidate: MaterializedEvaluationCandidate,
    execution: CandidateExecutionRecord,
    request: HumanEvalSubmissionRequest,
    *,
    any_function_group: bool = False,
) -> _CandidateOutcome | _CandidateHarnessFailure:
    """Classify one candidate's benchmark outcome.

    ``any_function_group`` selects the within-candidate rule. Extraction keeps a
    solution and the helpers it was written beside in one candidate, and
    evaluation runs the complete suite once per top-level function, so requiring
    every group to pass scores a correct solution zero for the company it keeps.
    When set, the candidate passes if any one function group passes the complete
    suite; when unset, every group's cases must pass.
    """

    outcome = execution.outcome
    if isinstance(outcome, HarnessExecutionFailure | ExecutorExecutionFailure):
        attributed = failure_class_of(outcome)
        assert attributed is not None
        return _CandidateHarnessFailure(
            failure_type=outcome.failure_type,
            message=outcome.message,
            failure_class=attributed,
        )
    if isinstance(outcome, CandidateJobTerminated):
        return _CandidateOutcome(
            outcome=(
                SubmissionOutcome.TIMED_OUT
                if outcome.reason is CandidateTerminationReason.WALL_TIME
                else SubmissionOutcome.TESTS_FAILED
            )
        )
    if not isinstance(outcome, CandidateJobCompleted):
        return _CandidateHarnessFailure(
            failure_type=type(outcome).__name__,
            message="unsupported candidate execution outcome",
        )
    if isinstance(outcome.result.namespace, CandidateNamespaceFailure):
        return _CandidateOutcome(
            outcome=(
                SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
                if not _top_level_functions(candidate.source.source)
                else SubmissionOutcome.TESTS_FAILED
            )
        )
    matching_suites = tuple(
        suite
        for suite in outcome.result.suites
        if suite.question == request.scoring_profile.question
    )
    if len(matching_suites) != 1:
        return _CandidateHarnessFailure(
            failure_type="MissingHumanEvalSuite",
            message=(
                "candidate result does not contain exactly the HumanEval "
                "scoring-profile suite"
            ),
        )
    suite = matching_suites[0]
    if isinstance(suite, HumanEvalSuiteHarnessFailure):
        return _CandidateHarnessFailure(
            failure_type=suite.failure_type,
            message=suite.message,
        )
    assert isinstance(suite, HumanEvalSuiteCompleted)
    cases = tuple(case for group in suite.groups for case in group.cases)
    if not outcome.result.namespace.function_names:
        return _CandidateOutcome(
            outcome=SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS
        )
    if any_function_group and any(
        group.cases
        and all(
            case.status is EvaluationCaseStatus.PASSED for case in group.cases
        )
        for group in suite.groups
    ):
        return _CandidateOutcome(outcome=SubmissionOutcome.PASSED)
    if any(case.status is EvaluationCaseStatus.TIMEOUT for case in cases):
        # The candidate exhausted its wall-time budget, so every group that did
        # not pass is an unfinished measurement rather than a measured failure.
        projected = SubmissionOutcome.TIMED_OUT
    elif not cases:
        projected = SubmissionOutcome.EVALUATION_INCOMPLETE
    elif all(case.status is EvaluationCaseStatus.PASSED for case in cases):
        projected = SubmissionOutcome.PASSED
    else:
        projected = SubmissionOutcome.TESTS_FAILED
    return _CandidateOutcome(outcome=projected)


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
    failure_class: FailureClass = FailureClass.HARNESS,
) -> HarnessFailure:
    return HarnessFailure(
        sample=request.sample,
        evaluation=None,
        cause=HarnessFailureCause(
            exception_type=failure_type,
            message=message,
        ),
        failure_class=failure_class,
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
