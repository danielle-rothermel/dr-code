"""Keep only candidates that compile."""

from __future__ import annotations

from typing import ClassVar

from dr_code.code_analysis import validate_python_source_with_ast
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
)


class FilterCompilable(Step):
    """Keep candidates where ``validate_python_source_with_ast`` compiles.

    Rejections become facts (``{"rejected_0": "SyntaxError: ..."}``) —
    recorded facts, not quality judgments. An empty survivor set is data;
    the absence surfaces at ``select_first``.
    """

    NAME: ClassVar[StepName] = StepName.FILTER_COMPILABLE
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        survivors: list[str] = []
        facts: dict[str, str] = {}
        for index, candidate in enumerate(value.candidates):
            validated = validate_python_source_with_ast(candidate)
            if validated.validation.compile_ok:
                survivors.append(candidate)
            else:
                reason = (
                    validated.validation.compile_error
                    or "candidate does not compile"
                )
                facts[f"rejected_{index}"] = reason
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


__all__ = ["FilterCompilable"]
