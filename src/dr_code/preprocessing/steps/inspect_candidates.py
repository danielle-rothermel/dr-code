from __future__ import annotations

import ast
from typing import ClassVar

from dr_code.core.source.python_analysis import validate_python_source_with_ast
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
    CandidateInspection,
    InspectedCodeCandidate,
    InspectedCodeCandidateSetArtifact,
)


def top_level_function_names(tree: ast.Module) -> tuple[str, ...]:
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def inspect_source(source: str) -> CandidateInspection:
    validated = validate_python_source_with_ast(source)
    validation = validated.validation
    return CandidateInspection(
        parses=validation.parse_ok,
        parse_error=validation.parse_error,
        compiles=validation.compile_ok,
        compile_error=validation.compile_error,
        top_level_function_names=(
            top_level_function_names(validated.tree)
            if validated.tree is not None
            else ()
        ),
    )


class InspectCandidates(Step[StepSettings]):
    NAME: ClassVar[StepName] = StepName.INSPECT_CANDIDATES
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        inspections: dict[str, CandidateInspection] = {}
        inspected: list[InspectedCodeCandidate] = []
        for candidate in candidate_set(value).candidates:
            inspection = inspections.get(candidate.source)
            if inspection is None:
                inspection = inspect_source(candidate.source)
                inspections[candidate.source] = inspection
            inspected.append(
                InspectedCodeCandidate(
                    candidate=candidate, inspection=inspection
                )
            )
        return StepOutput(
            value=InspectedCodeCandidateSetArtifact(
                candidates=tuple(inspected)
            ),
            facts={
                "inspected_count": len(inspected),
                "compiles_count": sum(
                    1 for item in inspected if item.inspection.compiles
                ),
            },
        )


__all__ = [
    "InspectCandidates",
    "inspect_source",
    "top_level_function_names",
]
