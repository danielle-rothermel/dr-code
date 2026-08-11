from __future__ import annotations

from collections.abc import Sequence

from dr_code.evaluation.identity import (
    EvaluationRuntimeIdentity,
    EvaluationSampleIdentity,
    EvaluationSlotIdentity,
)
from dr_code.evaluation.plan import EvaluationPlan
from dr_code.evaluation.records import (
    CandidateJobCompleted,
    EvaluationAttemptRecord,
    EvaluationMemberRecord,
    EvaluatedSampleRecord,
    PreprocessingAbsentSampleRecord,
    SampleEvaluationRecord,
)
from dr_code.metrics import (
    MetricQuestionCoordinate,
    MetricsDefinitionCoordinate,
)
from dr_code.preprocessing import bind_external_preprocessing
from dr_code.trace import (
    Absent,
    ExternalPreprocessingTraceProducer,
    PreprocessingTraceProducer,
)


def validate_evaluation_attempt_graph(
    attempt: EvaluationAttemptRecord,
    records: Sequence[SampleEvaluationRecord],
    /,
) -> None:
    """Validate one compact attempt and its ordered resolved evidence graph."""

    validate_attempt_membership(attempt)
    present = tuple(
        member for member in attempt.members if member.record is not None
    )
    if len(present) != len(records):
        raise ValueError(
            "resolved sample evidence does not cover every referenced member"
        )
    for member, record in zip(present, records, strict=True):
        validate_sample_record_graph(
            record,
            slot=member.slot,
            sample=member.sample,
            plan=attempt.plan,
            runtime=attempt.runtime,
            cache_namespace=attempt.cache_namespace,
        )


def validate_attempt_membership(attempt: EvaluationAttemptRecord, /) -> None:
    _validate_membership_order(attempt.plan, attempt.members)


def validate_sample_record_graph(
    record: SampleEvaluationRecord,
    /,
    *,
    slot: EvaluationSlotIdentity,
    sample: EvaluationSampleIdentity,
    plan: EvaluationPlan,
    runtime: EvaluationRuntimeIdentity,
    cache_namespace: str,
) -> None:
    """Validate all semantic links owned by one sample evidence record."""

    if record.slot != slot or record.sample.identity != sample:
        raise ValueError(
            "sample record identity does not match ordered membership"
        )
    if slot.task_id != record.sample.task_id:
        raise ValueError("sample record task does not match its slot")

    producer = record.trace.producer
    if not isinstance(
        producer,
        PreprocessingTraceProducer | ExternalPreprocessingTraceProducer,
    ):
        raise ValueError(
            "sample record trace must carry a preprocessing definition"
        )
    expected_producer = bind_external_preprocessing(
        plan.procedure.preprocessing
    ).producer
    if not isinstance(expected_producer, ExternalPreprocessingTraceProducer):
        raise AssertionError(
            "external preprocessing must preserve its coordinate"
        )
    expected_preprocessing = expected_producer.definition
    if producer.definition != expected_preprocessing:
        raise ValueError(
            "sample record trace preprocessing does not match the attempt plan"
        )

    if isinstance(record, PreprocessingAbsentSampleRecord):
        if record.trace.values.get("output") != record.absence:
            raise ValueError(
                "preprocessing absence must match the trace output"
            )
        return
    if not isinstance(record, EvaluatedSampleRecord):
        if isinstance(record.trace.values.get("output"), Absent):
            raise ValueError(
                "a no-candidates record cannot carry an absent trace output"
            )
        return

    candidate_identities = tuple(
        candidate.identity for candidate in record.candidates
    )
    expected_ordinals = tuple(range(len(candidate_identities)))
    if (
        tuple(identity.candidate_ordinal for identity in candidate_identities)
        != expected_ordinals
    ):
        raise ValueError(
            "materialized candidates must preserve contiguous ordinal order"
        )
    if any(
        identity.sample != record.sample.identity
        or identity.preprocessing != producer.definition
        for identity in candidate_identities
    ):
        raise ValueError(
            "materialized candidate identity does not match sample trace provenance"
        )

    for execution, candidate in zip(
        record.executions, candidate_identities, strict=True
    ):
        if execution.candidate != candidate:
            raise ValueError(
                "candidate execution does not match materialization order"
            )
        if execution.runtime != runtime:
            raise ValueError(
                "candidate execution runtime does not match the attempt"
            )
        if execution.cache_namespace != cache_namespace:
            raise ValueError(
                "candidate execution cache namespace does not match the attempt"
            )
        if isinstance(execution.outcome, CandidateJobCompleted) and (
            execution.outcome.result.candidate != candidate
        ):
            raise ValueError(
                "completed candidate result does not match its execution"
            )

    expected_questions = tuple(
        MetricQuestionCoordinate.of(question)
        for question in plan.procedure.metrics.questions
    )
    expected_definition = MetricsDefinitionCoordinate.of(
        plan.procedure.metrics
    )
    expected_count = len(record.candidates) * len(expected_questions)
    if len(record.metrics) != expected_count:
        raise ValueError(
            "metric evidence does not exactly cover planned candidate questions"
        )
    for candidate_index in range(len(record.candidates)):
        start = candidate_index * len(expected_questions)
        candidate_metrics = record.metrics[
            start : start + len(expected_questions)
        ]
        if (
            tuple(metric.identity.question for metric in candidate_metrics)
            != expected_questions
        ):
            raise ValueError(
                "metric evidence does not preserve planned question order"
            )
        if any(
            metric.identity.metrics_definition != expected_definition
            for metric in candidate_metrics
        ):
            raise ValueError(
                "metric evidence definition does not match the attempt plan"
            )
        if any(
            metric.identity.producer != record.trace.producer
            for metric in candidate_metrics
        ):
            raise ValueError(
                "metric evidence producer does not match the sample trace"
            )


def _validate_membership_order(
    plan: EvaluationPlan,
    members: Sequence[EvaluationMemberRecord],
) -> None:
    expected_slots = tuple(
        EvaluationSlotIdentity(
            task_set=plan.task_set.coordinate,
            repeat_plan=plan.repeat_plan.coordinate,
            task_id=task_id,
            repeat_index=repeat_index,
        )
        for task_id in plan.task_set.selected
        for repeat_index in range(plan.repeat_plan.repeats)
    )
    positions = {slot: index for index, slot in enumerate(expected_slots)}
    try:
        member_positions = tuple(positions[member.slot] for member in members)
    except KeyError as error:
        raise ValueError(
            "every evaluation member slot must belong to the evaluation plan"
        ) from error
    if member_positions != tuple(sorted(member_positions)):
        raise ValueError("evaluation members must preserve plan slot order")


__all__ = [
    "validate_attempt_membership",
    "validate_evaluation_attempt_graph",
    "validate_sample_record_graph",
]
