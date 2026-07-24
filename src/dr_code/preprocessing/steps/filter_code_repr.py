"""Drop candidates that are code-repr assignments."""

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


CODE_REPR_VARIABLE_NAME = "code"


def _is_code_repr_assignment(tree: ast.Module) -> bool:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return False
    statement = tree.body[0]
    if len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    return (
        isinstance(target, ast.Name)
        and target.id == CODE_REPR_VARIABLE_NAME
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


class FilterCodeRepr(Step):
    """Drop candidates shaped like ``code = \"...\"``."""

    NAME: ClassVar[StepName] = StepName.FILTER_CODE_REPR
    VERSION: ClassVar[str] = "2"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        survivors: list[str] = []
        lineage: list[CandidateLineage] = []
        survivor_records: list[dict[str, JsonValue]] = []
        rejections: list[dict[str, JsonValue]] = []

        for index, candidate in enumerate(value.candidates):
            validated = validate_python_source_with_ast(candidate)
            validation = validated.validation
            candidate_id = value.lineage_at(index).candidate_id
            if validated.tree is not None and _is_code_repr_assignment(
                validated.tree
            ):
                rejection: dict[str, JsonValue] = {
                    "input_index": index,
                    "reason_code": "code_repr_assignment",
                    "parse_ok": validation.parse_ok,
                    "parse_error": validation.parse_error,
                    "compile_ok": validation.compile_ok,
                    "compile_error": validation.compile_error,
                }
                if candidate_id is not None:
                    rejection["candidate_id"] = candidate_id
                rejections.append(rejection)
            else:
                survivors.append(candidate)
                survivor_record: dict[str, JsonValue] = {"input_index": index}
                if candidate_id is not None:
                    survivor_record["candidate_id"] = candidate_id
                survivor_records.append(survivor_record)
                if value.lineage:
                    lineage.append(value.lineage_at(index))

        facts: dict[str, JsonValue] = {
            "input_candidate_count": len(value.candidates),
            "survivor_candidate_count": len(survivors),
            "survivors": survivor_records,
            "rejections": rejections,
        }
        if not survivors:
            raise StepFailedError(
                "no candidate survived code-repr filtering",
                failure_code=PreprocessingFailureCode.CODE_REPR_ONLY,
                facts=facts,
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(survivors), lineage=tuple(lineage)
            ),
            facts=facts,
        )


__all__ = ["FilterCodeRepr"]
