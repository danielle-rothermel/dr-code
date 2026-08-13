from __future__ import annotations

import hashlib
from typing import Annotated, Literal, TypeAlias
from uuid import UUID

from dr_serialize import IdentityDocument, Sha256Digest
from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.coordinates import (
    DatasetCoordinate,
    SamplingPlanCoordinate,
    TaskSetCoordinate,
)
from dr_code.evaluation.references import EvidenceReference
from dr_code.evaluation.provenance import SyntheticSampleCoordinate
from dr_code.trace import (
    Artifact,
    CodeArtifact,
    PreprocessingDefinitionCoordinate,
    TextArtifact,
)


class EvalSlotId(FrozenModel):
    task_set: TaskSetCoordinate
    sampling_plan: SamplingPlanCoordinate
    task_id: str
    sample_index: int = Field(ge=0)


class EvalSourceId(FrozenModel):
    namespace: str = Field(min_length=1)
    value: str = Field(min_length=1)


class EvalSampleId(FrozenModel):
    sample_id: str = Field(min_length=1)


class EvalCandidateId(FrozenModel):
    """Identifies a candidate after materialization ordering is final."""

    sample: EvalSampleId
    preprocessing: PreprocessingDefinitionCoordinate
    candidate_ordinal: int = Field(ge=0)


class EvalRuntimeId(FrozenModel):
    document: IdentityDocument


class EvalAttemptId(FrozenModel):
    attempt_id: UUID


class GeneratedSampleProvenance(FrozenModel):
    kind: Literal["generated"] = "generated"
    source_identity: EvalSourceId
    source_reference: EvidenceReference
    generation_id: str


class CorpusSampleProvenance(FrozenModel):
    kind: Literal["corpus"] = "corpus"
    source_identity: EvalSourceId
    source_reference: EvidenceReference
    dataset: DatasetCoordinate
    row_id: str


class SyntheticSampleProvenance(FrozenModel):
    kind: Literal["synthetic"] = "synthetic"
    source_identity: EvalSourceId
    source_reference: EvidenceReference
    coordinate: SyntheticSampleCoordinate


EvalSampleProvenance: TypeAlias = Annotated[
    GeneratedSampleProvenance
    | CorpusSampleProvenance
    | SyntheticSampleProvenance,
    Field(discriminator="kind"),
]


class EvalSampleMetadata(FrozenModel):
    identity: EvalSampleId
    task_id: str
    provenance: EvalSampleProvenance


class EvalSampleAuxiliaryArtifact(FrozenModel):
    trace_key: str = Field(min_length=1)
    artifact: Artifact

    @model_validator(mode="after")
    def validate_trace_key(self) -> EvalSampleAuxiliaryArtifact:
        if self.trace_key in {"input", "output"}:
            raise ValueError(
                "sample auxiliary trace_key must not be input or output"
            )
        return self


class EvalSample(FrozenModel):
    metadata: EvalSampleMetadata
    raw_input: TextArtifact
    auxiliary_artifacts: tuple[EvalSampleAuxiliaryArtifact, ...] = ()

    @model_validator(mode="after")
    def validate_auxiliary_artifacts(self) -> EvalSample:
        keys = tuple(item.trace_key for item in self.auxiliary_artifacts)
        if len(set(keys)) != len(keys):
            raise ValueError("sample auxiliary trace keys must be unique")
        return self


class MaterializedEvalCandidate(FrozenModel):
    identity: EvalCandidateId
    source: CodeArtifact
    source_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_source_sha256(self) -> MaterializedEvalCandidate:
        actual = Sha256Digest(
            hashlib.sha256(self.source.source.encode("utf-8")).hexdigest()
        )
        if self.source_sha256 != actual:
            raise ValueError("source_sha256 must match the candidate source")
        return self


__all__ = [
    "CorpusSampleProvenance",
    "EvalAttemptId",
    "EvalCandidateId",
    "EvalRuntimeId",
    "EvalSample",
    "EvalSampleAuxiliaryArtifact",
    "EvalSampleId",
    "EvalSampleMetadata",
    "EvalSampleProvenance",
    "EvalSlotId",
    "EvalSourceId",
    "GeneratedSampleProvenance",
    "MaterializedEvalCandidate",
    "SyntheticSampleProvenance",
]
