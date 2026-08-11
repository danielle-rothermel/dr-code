from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from dr_serialize import Sha256Digest
from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.evaluation.aggregation import AggregationResult
from dr_code.evaluation.batch import ProjectionKind
from dr_code.evaluation.identity import (
    EvaluationAttemptIdentity,
    EvaluationCandidateIdentity,
    EvaluationSampleMetadata,
    EvaluationSlotIdentity,
)
from dr_code.evaluation.plan import AggregationPolicy
from dr_code.evaluation.references import EvidenceReference
from dr_code.evaluation.score import Score
from dr_code.metrics import (
    MetricQuestionCoordinate,
    MetricValue,
    RecordStatus,
)


class EvaluationSampleProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.EVALUATION_SAMPLES] = (
        ProjectionKind.EVALUATION_SAMPLES
    )
    source_attempt: EvaluationAttemptIdentity
    slot: EvaluationSlotIdentity
    sample: EvaluationSampleMetadata
    status: str
    record: EvidenceReference


class MaterializedCandidateProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.MATERIALIZED_CANDIDATES] = (
        ProjectionKind.MATERIALIZED_CANDIDATES
    )
    source_attempt: EvaluationAttemptIdentity
    candidate: EvaluationCandidateIdentity
    source_sha256: Sha256Digest
    sample_record: EvidenceReference


class MetricRecordProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.METRIC_RECORDS] = (
        ProjectionKind.METRIC_RECORDS
    )
    source_attempt: EvaluationAttemptIdentity
    candidate: EvaluationCandidateIdentity
    question: MetricQuestionCoordinate
    status: RecordStatus
    values: tuple[MetricValue, ...]
    sample_record: EvidenceReference


class AggregationResultProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.AGGREGATION_RESULTS] = (
        ProjectionKind.AGGREGATION_RESULTS
    )
    source_attempt: EvaluationAttemptIdentity
    policy: AggregationPolicy
    result: AggregationResult


class ScoreProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.SCORES] = ProjectionKind.SCORES
    source_attempt: EvaluationAttemptIdentity
    score: Score


ProjectionRow: TypeAlias = Annotated[
    EvaluationSampleProjectionRow
    | MaterializedCandidateProjectionRow
    | MetricRecordProjectionRow
    | AggregationResultProjectionRow
    | ScoreProjectionRow,
    Field(discriminator="kind"),
]


__all__ = [
    "AggregationResultProjectionRow",
    "EvaluationSampleProjectionRow",
    "MaterializedCandidateProjectionRow",
    "MetricRecordProjectionRow",
    "ProjectionRow",
    "ScoreProjectionRow",
]
