"""Convert internal identified candidates to the public candidate artifact."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
    IdentifiedCandidateSetArtifact,
)


class MaterializeCandidates(Step):
    """Drop stored inspections while preserving source, identity, and paths."""

    NAME: ClassVar[StepName] = StepName.MATERIALIZE_CANDIDATES
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, IdentifiedCandidateSetArtifact)
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(item.source for item in value.candidates),
                lineage=tuple(item.lineage for item in value.candidates),
            ),
            facts={"candidate_count": len(value.candidates)},
        )


__all__ = ["MaterializeCandidates"]
