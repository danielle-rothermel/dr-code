"""Return all candidates unchanged (cardinality knob)."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
)


class ReturnAll(Step):
    """Keep the complete ordered ``CodeCandidateSetArtifact`` unchanged."""

    NAME: ClassVar[StepName] = StepName.RETURN_ALL
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        if not value.candidates:
            raise StepFailedError(
                "no candidate available to return",
                failure_code=PreprocessingFailureCode.NO_CANDIDATES_TO_RETURN,
                facts={"candidate_count": 0},
            )
        return StepOutput(
            value=value,
            facts={
                "outcome_code": "function_candidates_extracted",
                "candidate_count": len(value.candidates),
            },
        )


__all__ = ["ReturnAll"]
