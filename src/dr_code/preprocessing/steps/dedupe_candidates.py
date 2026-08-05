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
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
)


class DedupeCandidates(Step[StepSettings]):
    NAME: ClassVar[StepName] = StepName.DEDUPE_CANDIDATES
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        candidates = candidate_set(value).candidates
        position: dict[str, int] = {}
        survivors: list[CodeCandidate] = []
        merged: list[list[CandidateOrigin]] = []
        for candidate in candidates:
            index = position.get(candidate.source)
            if index is None:
                position[candidate.source] = len(survivors)
                survivors.append(candidate)
                merged.append(list(candidate.origins))
            else:
                merged[index].extend(candidate.origins)
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(
                    CodeCandidate(
                        source=candidate.source, origins=tuple(origins)
                    )
                    for candidate, origins in zip(
                        survivors, merged, strict=True
                    )
                )
            ),
            facts={"duplicates_merged": len(candidates) - len(survivors)},
        )


__all__ = ["DedupeCandidates"]
