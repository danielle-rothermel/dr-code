"""Parse and compile each candidate once, pairing it with its inspection.

This is the only step that parses candidate sources. Every filter after it
reads the stored inspection instead of reparsing, so a candidate is parsed
exactly once no matter how many structural questions are asked about it.

The inspection describes the exact source it accompanies, which is why no
step after this one may rewrite a candidate's source: doing so would leave
the stored inspection describing text that is no longer there. Every
source-mutating step — cleaning, import inference, salvage — runs before
inspection for that reason.
"""

from __future__ import annotations

import ast
from typing import ClassVar

from dr_code.code_analysis import validate_python_source_with_ast
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
    """The names of ``tree``'s module-level function definitions, in order.

    Module level only: a method or a closure is not a top-level function,
    so nested definitions are deliberately not walked.
    """
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )


def inspect_source(source: str) -> CandidateInspection:
    """Parse and compile ``source`` once, reusing the tree it produced."""
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
    """CandidateSet -> InspectedCandidateSet, one parse per distinct source.

    The set reaching this step is already deduplicated, so every source is
    distinct; inspections are nonetheless memoized on the source text so
    the one-parse-per-source guarantee holds regardless of what precedes
    the step. Candidate order and lineage are carried through untouched.
    """

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
