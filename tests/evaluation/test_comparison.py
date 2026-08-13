from __future__ import annotations

import hashlib
from uuid import UUID

import pytest
from dr_serialize import Sha256Digest, canonical_json_bytes

from _builders import evaluation_slot, measured, sample_identity
from dr_code.evaluation import (
    BundleRecordReference,
    ComparableProjectionComparison,
    ComparisonStatus,
    EvalAttemptIdentity,
    EvalMemberRecord,
    EvaluatedSampleRecord,
    NoCandidatesSampleRecord,
    PreprocessingAbsentSampleRecord,
    ProjectionKind,
    ProjectionNotComparable,
    compare_eval_attempts,
)
from dr_code.trace import Absent, CodeArtifact, SerializedTrace

from .test_record_models import (
    attempt,
    evaluated,
    materialized,
    metadata,
    record_identity,
    trace,
)

pytestmark = pytest.mark.asyncio


class Resolver:
    def __init__(self, records=None):  # type: ignore[no-untyped-def]
        self.records = records or {}
        self.calls: list[BundleRecordReference] = []

    async def resolve(self, reference):  # type: ignore[no-untyped-def]
        self.calls.append(reference)
        return self.records[reference]


def _reference(
    record: (
        NoCandidatesSampleRecord
        | PreprocessingAbsentSampleRecord
        | EvaluatedSampleRecord
    ),
    *,
    artifact: str,
    index: int = 0,
) -> BundleRecordReference:
    payload = canonical_json_bytes(record.model_dump(mode="json"))
    return BundleRecordReference(
        artifact_name=artifact,
        record_index=index,
        record_sha256=Sha256Digest(hashlib.sha256(payload).hexdigest()),
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=1,
    )


def _record(index: int, *, output: str = "same") -> NoCandidatesSampleRecord:
    return NoCandidatesSampleRecord(
        slot=evaluation_slot(sample_index=index),
        sample=metadata(identity=sample_identity(sample_id=f"sample-{index}")),
        trace=SerializedTrace(
            schema_version=3,
            producer=record_identity().producer,
            values={
                "input": trace().values["input"],
                "output": CodeArtifact(source=output),
            },
        ),
    )


def _member(index: int, reference: BundleRecordReference) -> EvalMemberRecord:
    return EvalMemberRecord(
        slot=evaluation_slot(sample_index=index),
        sample=sample_identity(sample_id=f"sample-{index}"),
        record=reference,
    )


def _attempt(identity: int, members: tuple[EvalMemberRecord, ...]):
    base = attempt()
    plan = base.plan.model_copy(
        update={
            "sampling_plan": base.plan.sampling_plan.model_copy(
                update={"task_num_samples": (4, 4)}
            )
        }
    )
    return base.model_copy(
        update={
            "identity": EvalAttemptIdentity(attempt_id=UUID(int=identity)),
            "plan": plan,
            "members": members,
        }
    )


async def test_comparison_preserves_member_identity_and_reports_add_remove() -> (
    None
):
    records = tuple(_record(index) for index in range(4))
    references = tuple(
        _reference(record, artifact=f"record-{index}.json")
        for index, record in enumerate(records)
    )
    left = _attempt(
        10, tuple(_member(index, references[index]) for index in (0, 1, 2))
    )
    right = _attempt(
        11, tuple(_member(index, references[index]) for index in (0, 1, 3))
    )
    resolver = Resolver()

    result = await compare_eval_attempts(left, right, resolver=resolver)

    assert [item.identity.sample.sample_id for item in result.matched] == [
        "sample-0",
        "sample-1",
    ]
    assert [item.sample.sample_id for item in result.added] == ["sample-3"]
    assert [item.sample.sample_id for item in result.removed] == ["sample-2"]
    assert result.ordering_changed is False
    assert all(
        item.sample is ComparisonStatus.UNCHANGED
        and item.trace is ComparisonStatus.UNCHANGED
        and item.candidates is ComparisonStatus.UNCHANGED
        and item.metrics is ComparisonStatus.UNCHANGED
        for item in result.matched
    )
    assert resolver.calls == []


async def test_equal_content_hashes_short_circuit_without_resolution() -> None:
    record = _record(0)
    left_reference = _reference(record, artifact="left.json")
    right_reference = _reference(record, artifact="right.json", index=7)
    resolver = Resolver()

    result = await compare_eval_attempts(
        _attempt(10, (_member(0, left_reference),)),
        _attempt(11, (_member(0, right_reference),)),
        resolver=resolver,
    )

    assert result.matched[0].trace is ComparisonStatus.UNCHANGED
    assert resolver.calls == []


async def test_changed_matches_resolve_left_then_right_once_per_pair() -> None:
    left_records = (_record(0, output="left-0"), _record(1, output="left-1"))
    right_records = (
        _record(0, output="right-0"),
        _record(1, output="right-1"),
    )
    left_references = tuple(
        _reference(record, artifact=f"left-{index}.json")
        for index, record in enumerate(left_records)
    )
    right_references = tuple(
        _reference(record, artifact=f"right-{index}.json")
        for index, record in enumerate(right_records)
    )
    resolver = Resolver(
        {
            **dict(zip(left_references, left_records, strict=True)),
            **dict(zip(right_references, right_records, strict=True)),
        }
    )

    result = await compare_eval_attempts(
        _attempt(10, tuple(_member(i, left_references[i]) for i in range(2))),
        _attempt(11, tuple(_member(i, right_references[i]) for i in range(2))),
        resolver=resolver,
    )

    assert resolver.calls == [
        left_references[0],
        right_references[0],
        left_references[1],
        right_references[1],
    ]
    assert [item.trace for item in result.matched] == [
        ComparisonStatus.CHANGED,
        ComparisonStatus.CHANGED,
    ]


async def test_projection_comparison_reports_denominators_and_not_comparable() -> (
    None
):
    left_record = _record(0)
    absence = Absent(
        failed_step="extract",
        failure_code="none",
        cause="no candidate",
    )
    right_record = PreprocessingAbsentSampleRecord(
        slot=left_record.slot,
        sample=left_record.sample,
        trace=left_record.trace.model_copy(
            update={"values": {**left_record.trace.values, "output": absence}}
        ),
        absence=absence,
    )
    left_reference = _reference(left_record, artifact="left.json")
    right_reference = _reference(right_record, artifact="right.json")
    resolver = Resolver(
        {left_reference: left_record, right_reference: right_record}
    )

    result = await compare_eval_attempts(
        _attempt(10, (_member(0, left_reference),)),
        _attempt(11, (_member(0, right_reference),)),
        resolver=resolver,
        projections=(
            (ProjectionKind.EVAL_SAMPLES, 2, 2),
            (ProjectionKind.MATERIALIZED_CANDIDATES, 2, 2),
            (ProjectionKind.METRIC_RECORDS, 2, None),
            (ProjectionKind.SCORES, 2, 3),
        ),
    )

    samples = result.projections[0]
    candidates = result.projections[1]
    assert isinstance(samples, ComparableProjectionComparison)
    assert (
        samples.population,
        samples.available_denominator,
        samples.changed,
    ) == (
        1,
        1,
        1,
    )
    assert isinstance(candidates, ComparableProjectionComparison)
    assert candidates.changed == 0
    assert isinstance(result.projections[2], ProjectionNotComparable)
    assert isinstance(result.projections[3], ProjectionNotComparable)
    assert len(resolver.calls) == 2


async def test_comparison_reports_exact_candidate_and_metric_statuses() -> (
    None
):
    left_record = evaluated()
    changed_source = CodeArtifact(source="def f(): return 2")
    right_record = left_record.model_copy(
        update={
            "candidates": (
                materialized(
                    source=changed_source,
                    source_sha256=Sha256Digest(
                        hashlib.sha256(
                            changed_source.source.encode("utf-8")
                        ).hexdigest()
                    ),
                ),
            ),
            "metrics": (measured(2),),
        }
    )
    left_reference = _reference(left_record, artifact="left-evaluated.json")
    right_reference = _reference(right_record, artifact="right-evaluated.json")
    resolver = Resolver(
        {left_reference: left_record, right_reference: right_record}
    )

    result = await compare_eval_attempts(
        _attempt(10, (_member(0, left_reference),)),
        _attempt(11, (_member(0, right_reference),)),
        resolver=resolver,
    )

    (matched,) = result.matched
    assert matched.sample is ComparisonStatus.UNCHANGED
    assert matched.trace is ComparisonStatus.UNCHANGED
    assert matched.candidates is ComparisonStatus.CHANGED
    assert matched.metrics is ComparisonStatus.CHANGED


async def test_duplicate_semantic_membership_is_rejected() -> None:
    record = _record(0)
    reference = _reference(record, artifact="record.json")
    member = _member(0, reference)
    malformed = _attempt(10, (member, member))

    with pytest.raises(ValueError, match="duplicate semantic"):
        await compare_eval_attempts(
            malformed,
            _attempt(11, (member,)),
            resolver=Resolver(),
        )
