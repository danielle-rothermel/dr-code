"""Base for filters that keep or drop candidates from stored inspection.

A filter never transforms a candidate and never reparses one: it asks a
policy question of the inspection already stored alongside the source, and
records the reason for each rejection as a fact. Because filters do not
touch sources, a survivor's candidate record and its inspection are carried
through identically — the inspection still describes the exact source it
accompanies.
"""

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
    """Narrow an Artifact to its inspected-candidate-set member."""
    assert isinstance(value, InspectedCodeCandidateSetArtifact)
    return value


class InspectedFilterStep(Step[StepSettings]):
    """Elementwise keep/drop over inspected candidates.

    Subclasses implement ``rejection_reason``, returning the reason a
    candidate is dropped or ``None`` to keep it. Reasons are recorded as
    ``rejected_<ordinal>`` facts, where the ordinal indexes this step's
    *input* set.
    """

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
