from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from dr_serialize import Sha256Digest
from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.evaluation.aggregation import AggregationResult
from dr_code.evaluation.batch import ProjectionKind
from dr_code.evaluation.id import (
    EvalAttemptId,
    EvalCandidateId,
    EvalSampleMetadata,
    EvalSlotId,
)
from dr_code.evaluation.plan import AggregationPolicy
from dr_code.evaluation.references import EvidenceReference
from dr_code.evaluation.score import Score
from dr_code.metrics import (
    MetricQuestionCoordinate,
    MetricValue,
    RecordStatus,
)


class EvalSampleProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.EVAL_SAMPLES] = ProjectionKind.EVAL_SAMPLES
    source_attempt: EvalAttemptId
    slot: EvalSlotId
    sample: EvalSampleMetadata
    status: str
    record: EvidenceReference


class MaterializedCandidateProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.MATERIALIZED_CANDIDATES] = (
        ProjectionKind.MATERIALIZED_CANDIDATES
    )
    source_attempt: EvalAttemptId
    candidate: EvalCandidateId
    source_sha256: Sha256Digest
    sample_record: EvidenceReference


class MetricRecordProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.METRIC_RECORDS] = (
        ProjectionKind.METRIC_RECORDS
    )
    source_attempt: EvalAttemptId
    candidate: EvalCandidateId
    question: MetricQuestionCoordinate
    status: RecordStatus
    values: tuple[MetricValue, ...]
    sample_record: EvidenceReference


class AggregationResultProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.AGGREGATION_RESULTS] = (
        ProjectionKind.AGGREGATION_RESULTS
    )
    source_attempt: EvalAttemptId
    policy: AggregationPolicy
    result: AggregationResult


class ScoreProjectionRow(FrozenModel):
    kind: Literal[ProjectionKind.SCORES] = ProjectionKind.SCORES
    source_attempt: EvalAttemptId
    score: Score


ProjectionRow: TypeAlias = Annotated[
    EvalSampleProjectionRow
    | MaterializedCandidateProjectionRow
    | MetricRecordProjectionRow
    | AggregationResultProjectionRow
    | ScoreProjectionRow,
    Field(discriminator="kind"),
]


__all__ = [
    "AggregationResultProjectionRow",
    "EvalSampleProjectionRow",
    "MaterializedCandidateProjectionRow",
    "MetricRecordProjectionRow",
    "ProjectionRow",
    "ScoreProjectionRow",
]
