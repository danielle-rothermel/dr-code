from __future__ import annotations

from typing import ClassVar

from dr_code.core.source.text_transforms import drop_after_last_return
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
    ExtractionOperation,
)


class AddLastReturnSalvage(Step[StepSettings]):
    NAME: ClassVar[StepName] = StepName.ADD_LAST_RETURN_SALVAGE
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        operation = ExtractionOperation(operation_name=self.NAME.value)
        result: list[CodeCandidate] = []
        salvaged = 0
        for index, candidate in enumerate(candidate_set(value).candidates):
            result.append(candidate)
            truncated = drop_after_last_return(candidate.source)
            if truncated is None or truncated == candidate.source:
                continue
            result.append(
                candidate.extended(
                    CandidateOrigin(operation=operation, input_location=index),
                    source=truncated,
                )
            )
            salvaged += 1
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(result)),
            facts={"salvaged_count": salvaged},
        )


__all__ = ["AddLastReturnSalvage"]
