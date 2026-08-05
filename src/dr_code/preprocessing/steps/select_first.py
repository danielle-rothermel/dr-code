"""Select the first surviving candidate as Code."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    FailureCode,
    Step,
    StepFailedError,
    StepOutput,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
)


class SelectFirst(Step):
    """Deliberately dumb cardinality knob.

    First surviving candidate wins, preserving the ladder's
    conservative-first ordering. An empty set raises
    ``StepFailedError`` — the absence surfaces here, where cardinality is
    fixed.
    """

    NAME: ClassVar[StepName] = StepName.SELECT_FIRST
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        candidates = value.candidates
        if not candidates:
            raise StepFailedError(
                FailureCode.NO_CANDIDATE_SURVIVED_FILTERING,
                "no candidate survived filtering",
            )
        return StepOutput(value=CodeArtifact(source=candidates[0].source))


__all__ = ["SelectFirst"]
