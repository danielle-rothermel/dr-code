from __future__ import annotations

import hashlib
from typing import Annotated, Literal, TypeAlias
from uuid import UUID

from dr_serialize import IdentityDocument, Sha256Digest
from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.coordinates import (
    DatasetCoordinate,
    RepeatPlan,
    RepeatPlanCoordinate,
    TaskSetCoordinate,
)
from dr_code.evaluation.references import EvidenceReference
from dr_code.synthetic.models import SyntheticSampleCoordinate
from dr_code.trace import (
    Artifact,
    CodeArtifact,
    PreprocessingDefinitionCoordinate,
    TextArtifact,
)


class EvaluationSlotIdentity(FrozenModel):
    task_set: TaskSetCoordinate
    repeat_plan: RepeatPlanCoordinate
    task_id: str
    repeat_index: int = Field(ge=0)

    def within(self, plan: RepeatPlan) -> bool:
        return (
            self.repeat_plan == plan.coordinate
            and 0 <= self.repeat_index < plan.repeats
        )


class EvaluationSourceIdentity(FrozenModel):
    namespace: str = Field(min_length=1)
    value: str = Field(min_length=1)


class EvaluationSampleIdentity(FrozenModel):
    sample_id: str = Field(min_length=1)


class EvaluationCandidateIdentity(FrozenModel):
    """Identifies a candidate after materialization ordering is final."""

    sample: EvaluationSampleIdentity
    preprocessing: PreprocessingDefinitionCoordinate
    candidate_ordinal: int = Field(ge=0)


class EvaluationRuntimeIdentity(FrozenModel):
    document: IdentityDocument


class EvaluationAttemptIdentity(FrozenModel):
    attempt_id: UUID


class GeneratedSampleProvenance(FrozenModel):
    kind: Literal["generated"] = "generated"
    source_identity: EvaluationSourceIdentity
    source_reference: EvidenceReference
    generation_id: str


class CorpusSampleProvenance(FrozenModel):
    kind: Literal["corpus"] = "corpus"
    source_identity: EvaluationSourceIdentity
    source_reference: EvidenceReference
    dataset: DatasetCoordinate
    row_id: str


class SyntheticSampleProvenance(FrozenModel):
    kind: Literal["synthetic"] = "synthetic"
    source_identity: EvaluationSourceIdentity
    source_reference: EvidenceReference
    coordinate: SyntheticSampleCoordinate


EvaluationSampleProvenance: TypeAlias = Annotated[
    GeneratedSampleProvenance
    | CorpusSampleProvenance
    | SyntheticSampleProvenance,
    Field(discriminator="kind"),
]


class EvaluationSampleMetadata(FrozenModel):
    identity: EvaluationSampleIdentity
    task_id: str
    provenance: EvaluationSampleProvenance


class EvaluationSampleAuxiliaryArtifact(FrozenModel):
    trace_key: str = Field(min_length=1)
    artifact: Artifact

    @model_validator(mode="after")
    def validate_trace_key(self) -> EvaluationSampleAuxiliaryArtifact:
        if self.trace_key in {"input", "output"}:
            raise ValueError(
                "sample auxiliary trace_key must not be input or output"
            )
        return self


class EvaluationSample(FrozenModel):
    metadata: EvaluationSampleMetadata
    raw_input: TextArtifact
    auxiliary_artifacts: tuple[EvaluationSampleAuxiliaryArtifact, ...] = ()

    @model_validator(mode="after")
    def validate_auxiliary_artifacts(self) -> EvaluationSample:
        keys = tuple(item.trace_key for item in self.auxiliary_artifacts)
        if len(set(keys)) != len(keys):
            raise ValueError("sample auxiliary trace keys must be unique")
        return self


class MaterializedEvaluationCandidate(FrozenModel):
    identity: EvaluationCandidateIdentity
    source: CodeArtifact
    source_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_source_sha256(self) -> MaterializedEvaluationCandidate:
        actual = Sha256Digest(
            hashlib.sha256(self.source.source.encode("utf-8")).hexdigest()
        )
        if self.source_sha256 != actual:
            raise ValueError("source_sha256 must match the candidate source")
        return self


__all__ = [
    "CorpusSampleProvenance",
    "EvaluationAttemptIdentity",
    "EvaluationCandidateIdentity",
    "EvaluationRuntimeIdentity",
    "EvaluationSample",
    "EvaluationSampleAuxiliaryArtifact",
    "EvaluationSampleIdentity",
    "EvaluationSampleMetadata",
    "EvaluationSampleProvenance",
    "EvaluationSlotIdentity",
    "EvaluationSourceIdentity",
    "GeneratedSampleProvenance",
    "MaterializedEvaluationCandidate",
    "SyntheticSampleProvenance",
]
