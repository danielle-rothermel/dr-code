"""Keep candidates that define a function at module scope."""

from __future__ import annotations

import ast
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


class FilterHasTopLevelFunction(Step):
    """Keep candidates with at least one top-level sync or async function."""

    NAME: ClassVar[StepName] = StepName.FILTER_HAS_TOP_LEVEL_FUNCTION
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        survivors: list[str] = []
        lineage: list[CandidateLineage] = []
        survivor_details: list[dict[str, JsonValue]] = []
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
            }
            candidate_id = value.lineage_at(index).candidate_id
            if candidate_id is not None:
                diagnostics["candidate_id"] = candidate_id
            if not validation.compile_ok:
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
                continue

            assert validated.tree is not None
            functions = [
                statement
                for statement in validated.tree.body
                if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            function_names = [function.name for function in functions]
            async_function_names = [
                function.name
                for function in functions
                if isinstance(function, ast.AsyncFunctionDef)
            ]
            function_details: dict[str, JsonValue] = {
                **diagnostics,
                "top_level_function_count": len(functions),
                "top_level_function_names": function_names,
                "top_level_async_function_names": async_function_names,
                "has_async_top_level_function": bool(async_function_names),
            }
            if not functions:
                rejections.append(
                    {
                        **function_details,
                        "reason_code": "no_top_level_function",
                    }
                )
                continue

            survivors.append(candidate)
            survivor_details.append(function_details)
            if value.lineage:
                lineage.append(value.lineage_at(index))

        facts: dict[str, JsonValue] = {
            "input_candidate_count": len(value.candidates),
            "survivor_candidate_count": len(survivors),
            "survivors": survivor_details,
            "rejections": rejections,
        }
        if not survivors:
            raise StepFailedError(
                "no candidate defined a top-level function",
                failure_code=(
                    PreprocessingFailureCode.NO_TOP_LEVEL_FUNCTION_CANDIDATE
                ),
                facts=facts,
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(survivors), lineage=tuple(lineage)
            ),
            facts=facts,
        )


__all__ = ["FilterHasTopLevelFunction"]
