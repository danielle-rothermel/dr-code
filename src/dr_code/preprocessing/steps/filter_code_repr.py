"""Drop candidates that are code-repr assignments."""

from __future__ import annotations

from typing import ClassVar

from dr_code.humaneval.code_parsing import is_code_repr_assignment
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
)


class FilterCodeRepr(Step):
    """Drop candidates that are ``code = "..."`` repr assignments.

    Wraps ``code_parsing.is_code_repr_assignment``. Rejections are
    recorded as facts.
    """

    NAME: ClassVar[StepName] = StepName.FILTER_CODE_REPR
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        survivors: list[str] = []
        facts: dict[str, str] = {}
        for index, candidate in enumerate(value.candidates):
            if is_code_repr_assignment(candidate):
                facts[f"rejected_{index}"] = "code repr assignment"
            else:
                survivors.append(candidate)
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


__all__ = ["FilterCodeRepr"]
