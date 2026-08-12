from __future__ import annotations

import hashlib
from enum import StrEnum, UNIQUE, verify
from typing import Annotated, Literal, Protocol, TypeAlias

from dr_serialize import canonical_json_bytes
from dr_store import ContentHashMismatchError
from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.batch import ProjectionKind
from dr_code.evaluation.identity import (
    EvaluationAttemptIdentity,
    EvaluationSampleIdentity,
    EvaluationSlotIdentity,
)
from dr_code.evaluation.records import (
    EvaluationAttemptRecord,
    EvaluationMemberRecord,
    EvaluatedSampleRecord,
    SampleEvaluationRecord,
)
from dr_code.evaluation.references import (
    BundleRecordReference,
    EvidenceReference,
)


class EvaluationEvidenceResolver(Protocol):
    async def resolve(
        self, reference: EvidenceReference, /
    ) -> SampleEvaluationRecord: ...


@verify(UNIQUE)
class ComparisonStatus(StrEnum):
    UNCHANGED = "unchanged"
    CHANGED = "changed"


class StructuralMemberIdentity(FrozenModel):
    slot: EvaluationSlotIdentity
    sample: EvaluationSampleIdentity


class StructuralRecordComparison(FrozenModel):
    identity: StructuralMemberIdentity
    left: EvidenceReference
    right: EvidenceReference
    sample: ComparisonStatus
    trace: ComparisonStatus
    candidates: ComparisonStatus
    metrics: ComparisonStatus


class ComparableProjectionComparison(FrozenModel):
    kind: Literal["comparable"] = "comparable"
    projection: ProjectionKind
    definition_version: Literal[2] = 2
    population: int = Field(ge=0)
    available_denominator: int = Field(ge=0)
    changed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> ComparableProjectionComparison:
        if self.available_denominator > self.population:
            raise ValueError(
                "available denominator must not exceed population"
            )
        if self.changed > self.available_denominator:
            raise ValueError(
                "changed count must not exceed available denominator"
            )
        return self


class ProjectionNotComparable(FrozenModel):
    kind: Literal["not_comparable"] = "not_comparable"
    projection: ProjectionKind
    left_definition_version: int | None
    right_definition_version: int | None
    reason: str = Field(min_length=1)


ProjectionComparison: TypeAlias = Annotated[
    ComparableProjectionComparison | ProjectionNotComparable,
    Field(discriminator="kind"),
]


class StructuralEvaluationComparison(FrozenModel):
    left: EvaluationAttemptIdentity
    right: EvaluationAttemptIdentity
    matched: tuple[StructuralRecordComparison, ...]
    added: tuple[StructuralMemberIdentity, ...]
    removed: tuple[StructuralMemberIdentity, ...]
    ordering_changed: bool
    projections: tuple[ProjectionComparison, ...]


_ProjectionDefinition: TypeAlias = tuple[
    ProjectionKind, int | None, int | None
]
_MemberIdentity: TypeAlias = tuple[
    EvaluationSlotIdentity, EvaluationSampleIdentity
]


async def compare_evaluation_attempts(
    left: EvaluationAttemptRecord,
    right: EvaluationAttemptRecord,
    *,
    resolver: EvaluationEvidenceResolver,
    projections: tuple[_ProjectionDefinition, ...] = (),
) -> StructuralEvaluationComparison:
    """Compare compact attempts, resolving only changed matched evidence.

    Projection definitions are ``(kind, left_version, right_version)`` tuples.
    A missing version makes that requested projection explicitly unavailable.
    """

    from dr_code.evaluation.validation import validate_attempt_membership

    validate_attempt_membership(left)
    validate_attempt_membership(right)
    left_order, left_members = _index_members(left)
    right_order, right_members = _index_members(right)
    common = set(left_members) & set(right_members)
    removed = tuple(
        StructuralMemberIdentity(slot=identity[0], sample=identity[1])
        for identity in left_order
        if identity not in right_members
    )
    added = tuple(
        StructuralMemberIdentity(slot=identity[0], sample=identity[1])
        for identity in right_order
        if identity not in left_members
    )
    ordering_changed = tuple(
        identity for identity in left_order if identity in common
    ) != tuple(identity for identity in right_order if identity in common)

    projection_kinds = _validate_projection_definitions(projections)
    matched: list[StructuralRecordComparison] = []
    projection_changes = {kind: 0 for kind in projection_kinds}
    for identity in left_order:
        if identity not in common:
            continue
        left_member = left_members[identity]
        right_member = right_members[identity]
        if left_member.record is None or right_member.record is None:
            raise ValueError(
                "matched comparison members require evidence references"
            )
        if _references_have_equal_content(
            left_member.record, right_member.record
        ):
            statuses = _unchanged_component_statuses()
        else:
            left_record = await resolver.resolve(left_member.record)
            _validate_resolved_record(left_record, left_member, left)
            _verify_reference(left_member.record, left_record)
            right_record = await resolver.resolve(right_member.record)
            _validate_resolved_record(right_record, right_member, right)
            _verify_reference(right_member.record, right_record)
            statuses = _component_statuses(left_record, right_record)
            for kind in projection_changes:
                if kind in {
                    ProjectionKind.EVALUATION_SAMPLES,
                    ProjectionKind.MATERIALIZED_CANDIDATES,
                    ProjectionKind.METRIC_RECORDS,
                } and not _projection_values_equal(
                    kind, left_record, right_record
                ):
                    projection_changes[kind] += 1
        matched.append(
            StructuralRecordComparison(
                identity=StructuralMemberIdentity(
                    slot=left_member.slot,
                    sample=left_member.sample,
                ),
                left=left_member.record,
                right=right_member.record,
                sample=statuses[0],
                trace=statuses[1],
                candidates=statuses[2],
                metrics=statuses[3],
            )
        )

    projection_results = tuple(
        _compare_projection(
            definition_pair,
            kind=kind,
            population=len(left_members.keys() | right_members.keys()),
            matched=tuple(matched),
            changed=projection_changes[kind],
        )
        for definition_pair, kind in zip(
            projections, projection_kinds, strict=True
        )
    )
    return StructuralEvaluationComparison(
        left=left.identity,
        right=right.identity,
        matched=tuple(matched),
        added=added,
        removed=removed,
        ordering_changed=ordering_changed,
        projections=projection_results,
    )


def _index_members(
    attempt: EvaluationAttemptRecord,
) -> tuple[
    tuple[_MemberIdentity, ...],
    dict[_MemberIdentity, EvaluationMemberRecord],
]:
    order: list[_MemberIdentity] = []
    members: dict[_MemberIdentity, EvaluationMemberRecord] = {}
    for member in attempt.members:
        identity: _MemberIdentity = (member.slot, member.sample)
        if identity in members:
            raise ValueError(
                "attempt has duplicate semantic sample membership"
            )
        order.append(identity)
        members[identity] = member
    return tuple(order), members


def _validate_projection_definitions(
    projections: tuple[_ProjectionDefinition, ...],
) -> tuple[ProjectionKind, ...]:
    kinds: list[ProjectionKind] = []
    for kind, left_version, right_version in projections:
        if left_version is None and right_version is None:
            raise ValueError(
                "a projection comparison requires at least one definition"
            )
        for version in (left_version, right_version):
            if version is not None and (
                isinstance(version, bool) or version < 1
            ):
                raise ValueError(
                    "projection definition versions must be positive"
                )
        if kind in kinds:
            raise ValueError("projection comparison requests must be unique")
        kinds.append(kind)
    return tuple(kinds)


def _compare_projection(
    definition: _ProjectionDefinition,
    *,
    kind: ProjectionKind,
    population: int,
    matched: tuple[StructuralRecordComparison, ...],
    changed: int,
) -> ProjectionComparison:
    _, left_version, right_version = definition
    if left_version is None or right_version is None:
        return ProjectionNotComparable(
            projection=kind,
            left_definition_version=left_version,
            right_definition_version=right_version,
            reason="projection definition is missing from one attempt",
        )
    if left_version != right_version or left_version != 2:
        return ProjectionNotComparable(
            projection=kind,
            left_definition_version=left_version,
            right_definition_version=right_version,
            reason="projection definition versions do not match",
        )
    if kind in {
        ProjectionKind.AGGREGATION_RESULTS,
        ProjectionKind.SCORES,
    }:
        return ProjectionNotComparable(
            projection=kind,
            left_definition_version=left_version,
            right_definition_version=right_version,
            reason="projection rows are not present in structural evidence",
        )

    return ComparableProjectionComparison(
        projection=kind,
        population=population,
        available_denominator=len(matched),
        changed=changed,
    )


def _projection_values_equal(
    kind: ProjectionKind,
    left: SampleEvaluationRecord,
    right: SampleEvaluationRecord,
) -> bool:
    if kind is ProjectionKind.EVALUATION_SAMPLES:
        return left.status == right.status and left.sample == right.sample
    if kind is ProjectionKind.MATERIALIZED_CANDIDATES:
        return _candidates(left) == _candidates(right)
    if kind is ProjectionKind.METRIC_RECORDS:
        return _metrics(left) == _metrics(right)
    raise AssertionError(f"unsupported comparable projection: {kind}")


def _component_statuses(
    left: SampleEvaluationRecord,
    right: SampleEvaluationRecord,
) -> tuple[
    ComparisonStatus,
    ComparisonStatus,
    ComparisonStatus,
    ComparisonStatus,
]:
    return (
        _comparison_status(
            left.status == right.status and left.sample == right.sample
        ),
        _comparison_status(left.trace == right.trace),
        _comparison_status(_candidates(left) == _candidates(right)),
        _comparison_status(_metrics(left) == _metrics(right)),
    )


def _unchanged_component_statuses() -> tuple[
    ComparisonStatus,
    ComparisonStatus,
    ComparisonStatus,
    ComparisonStatus,
]:
    return (
        ComparisonStatus.UNCHANGED,
        ComparisonStatus.UNCHANGED,
        ComparisonStatus.UNCHANGED,
        ComparisonStatus.UNCHANGED,
    )


def _comparison_status(equal: bool) -> ComparisonStatus:
    return ComparisonStatus.UNCHANGED if equal else ComparisonStatus.CHANGED


def _candidates(
    record: SampleEvaluationRecord,
) -> tuple[object, ...]:
    return (
        record.candidates if isinstance(record, EvaluatedSampleRecord) else ()
    )


def _metrics(record: SampleEvaluationRecord) -> tuple[object, ...]:
    return record.metrics if isinstance(record, EvaluatedSampleRecord) else ()


def _validate_resolved_record(
    record: SampleEvaluationRecord,
    member: EvaluationMemberRecord,
    attempt: EvaluationAttemptRecord,
) -> None:
    from dr_code.evaluation.validation import validate_sample_record_graph

    validate_sample_record_graph(
        record,
        slot=member.slot,
        sample=member.sample,
        plan=attempt.plan,
        runtime=attempt.runtime,
        cache_namespace=attempt.cache_namespace,
    )


def _references_have_equal_content(
    left: EvidenceReference, right: EvidenceReference
) -> bool:
    return left == right or _reference_content(left) == _reference_content(
        right
    )


def _reference_content(reference: EvidenceReference) -> tuple[str, int, str]:
    if isinstance(reference, BundleRecordReference):
        return (
            reference.schema,
            reference.schema_version,
            str(reference.record_sha256),
        )
    return (
        reference.reference.schema,
        reference.schema_version,
        reference.reference.content_hash,
    )


def _verify_reference(
    reference: EvidenceReference, record: SampleEvaluationRecord
) -> None:
    payload = record.model_dump(mode="json")
    if isinstance(reference, BundleRecordReference):
        actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        if actual != str(reference.record_sha256):
            raise ValueError(
                "resolved evidence does not match its content hash"
            )
    else:
        try:
            reference.reference.verify_record(payload)
        except ContentHashMismatchError as error:
            raise ValueError(
                "resolved evidence does not match its content hash"
            ) from error


__all__ = [
    "ComparableProjectionComparison",
    "ComparisonStatus",
    "EvaluationEvidenceResolver",
    "ProjectionComparison",
    "ProjectionNotComparable",
    "StructuralEvaluationComparison",
    "StructuralMemberIdentity",
    "StructuralRecordComparison",
    "compare_evaluation_attempts",
]
