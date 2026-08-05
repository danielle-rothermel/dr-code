"""Drop candidates that are plain literal modules."""

from __future__ import annotations

from typing import ClassVar

from dr_code.humaneval.code_parsing import is_plain_literal_module
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidate,
    CodeCandidateSetArtifact,
    JsonFactValue,
)


class FilterPlainLiteral(Step):
    """Drop candidates that are plain literal modules.

    Wraps ``code_parsing.is_plain_literal_module``. Rejections are
    recorded as facts.
    """

    NAME: ClassVar[StepName] = StepName.FILTER_PLAIN_LITERAL
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        survivors: list[CodeCandidate] = []
        facts: dict[str, JsonFactValue] = {}
        for index, candidate in enumerate(value.candidates):
            if is_plain_literal_module(candidate.source):
                facts[f"rejected_{index}"] = "plain literal module"
            else:
                survivors.append(candidate)
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


__all__ = ["FilterPlainLiteral"]
