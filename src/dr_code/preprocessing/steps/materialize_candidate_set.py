from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.preprocessing.steps.inspected_filter import (
    inspected_candidate_set,
)
from dr_code.trace import Artifact, ArtifactKind


class MaterializeCandidateSet(Step[StepSettings]):
    NAME: ClassVar[StepName] = StepName.MATERIALIZE_CANDIDATE_SET
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        materialized = inspected_candidate_set(value)
        if not materialized.candidates:
            raise StepFailedError(
                PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING,
                "no candidate survived filtering",
                evidence={"candidate_count": 0},
            )
        return StepOutput(
            value=materialized,
            facts={"candidate_count": len(materialized.candidates)},
        )


__all__ = ["MaterializeCandidateSet"]
