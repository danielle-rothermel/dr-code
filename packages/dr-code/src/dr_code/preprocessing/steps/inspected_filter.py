from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.steps.base import (
    Step,
    StepOutput,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    InspectedCodeCandidate,
    InspectedCodeCandidateSetArtifact,
    JsonFactValue,
)


def inspected_candidate_set(
    value: Artifact,
) -> InspectedCodeCandidateSetArtifact:
    assert isinstance(value, InspectedCodeCandidateSetArtifact)
    return value


class InspectedFilterStep(Step[StepSettings]):
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET

    def rejection_reason(
        self, inspected: InspectedCodeCandidate
    ) -> str | None:
        raise NotImplementedError

    def apply(self, value: Artifact) -> StepOutput:
        survivors: list[InspectedCodeCandidate] = []
        facts: dict[str, JsonFactValue] = {}
        for index, inspected in enumerate(
            inspected_candidate_set(value).candidates
        ):
            reason = self.rejection_reason(inspected)
            if reason is None:
                survivors.append(inspected)
            else:
                facts[f"rejected_{index}"] = reason
        return StepOutput(
            value=InspectedCodeCandidateSetArtifact(
                candidates=tuple(survivors)
            ),
            facts=facts,
        )


__all__ = ["InspectedFilterStep", "inspected_candidate_set"]
