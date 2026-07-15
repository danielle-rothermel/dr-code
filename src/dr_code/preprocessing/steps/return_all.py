"""Return all candidates unchanged (cardinality knob)."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import (
    ArtifactKind,
    CodeCandidateSetArtifact,
)


class ReturnAll(Step):
    """Keep all candidates; the deliberate "keep everything" cardinality knob.

    Pairs with ``select_first``: definitions choose whether to collapse to
    one ``Code`` or retain the full ``CodeCandidateSetArtifact``. Records
    the surviving candidate count as a fact.
    """

    NAME: ClassVar[StepName] = StepName.RETURN_ALL
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: CodeCandidateSetArtifact) -> StepOutput:
        return StepOutput(
            value=value,
            facts={"candidate_count": str(len(value.candidates))},
        )


__all__ = ["ReturnAll"]
