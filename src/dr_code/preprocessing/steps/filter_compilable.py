"""Keep only candidates that compile."""

from __future__ import annotations

from typing import ClassVar

from pydantic import JsonValue

from dr_code.code_analysis import validate_python_source_with_ast
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateLineage,
    CodeCandidateSetArtifact,
)


class FilterCompilable(Step):
    """Keep candidates that parse and compile as Python source."""

    NAME: ClassVar[StepName] = StepName.FILTER_COMPILABLE
    VERSION: ClassVar[str] = "3"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        survivors: list[str] = []
        lineage: list[CandidateLineage] = []
        survivor_validations: list[dict[str, JsonValue]] = []
        rejections: list[dict[str, JsonValue]] = []

        for index, candidate in enumerate(value.candidates):
            validated = validate_python_source_with_ast(candidate)
            validation = validated.validation
            diagnostics: dict[str, JsonValue] = {
                "input_index": index,
                "parse_ok": validation.parse_ok,
                "parse_error": validation.parse_error,
                "compile_ok": validation.compile_ok,
                "compile_error": validation.compile_error,
                "compile_warnings": list(validation.compile_warnings),
            }
            candidate_id = value.lineage_at(index).candidate_id
            if candidate_id is not None:
                diagnostics["candidate_id"] = candidate_id
            if validation.compile_ok:
                survivors.append(candidate)
                survivor_validations.append(diagnostics)
                if value.lineage:
                    lineage.append(value.lineage_at(index))
            else:
                rejections.append(
                    {
                        **diagnostics,
                        "reason_code": (
                            "parser_stack_overflow"
                            if validation.parser_stack_overflow
                            else (
                                "parser_recursion_overflow"
                                if validation.parser_recursion_overflow
                                else "not_compilable"
                            )
                        ),
                    }
                )

        facts: dict[str, JsonValue] = {
            "input_candidate_count": len(value.candidates),
            "survivor_candidate_count": len(survivors),
            "survivors": survivor_validations,
            "rejections": rejections,
        }
        if not survivors:
            raise StepFailedError(
                "no candidate compiled",
                failure_code=PreprocessingFailureCode.NO_COMPILABLE_CANDIDATE,
                facts=facts,
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(survivors), lineage=tuple(lineage)
            ),
            facts=facts,
        )


__all__ = ["FilterCompilable"]
