from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepOutput,
    StepSettings,
    candidate_set,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidate,
    CodeCandidateSetArtifact,
)


class DropBlankCandidates(Step[StepSettings]):
    NAME: ClassVar[StepName] = StepName.DROP_BLANK_CANDIDATES
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        candidates = candidate_set(value).candidates
        survivors: tuple[CodeCandidate, ...] = tuple(
            candidate for candidate in candidates if candidate.source.strip()
        )
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=survivors),
            facts={"dropped_count": len(candidates) - len(survivors)},
        )


__all__ = ["DropBlankCandidates"]
