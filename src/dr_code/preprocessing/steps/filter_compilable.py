"""Keep only identified candidates that compile."""

from __future__ import annotations

from typing import ClassVar

from pydantic import JsonValue

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.preprocessing.steps.filter_plain_literal import (
    _diagnostics,
    _record,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateInspection,
    IdentifiedCandidateSetArtifact,
)


class FilterCompilable(Step):
    """Keep candidates whose stored inspection compiled successfully."""

    NAME: ClassVar[StepName] = StepName.FILTER_COMPILABLE
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, IdentifiedCandidateSetArtifact)
        survivors = []
        survivor_records: list[dict[str, JsonValue]] = []
        rejections: list[dict[str, JsonValue]] = []
        for index, candidate in enumerate(value.candidates):
            inspection = candidate.inspection
            diagnostics = {
                **_record(index, candidate.lineage.candidate_id),
                **_diagnostics(inspection),
            }
            if inspection.compile_ok:
                survivors.append(candidate)
                survivor_records.append(diagnostics)
            else:
                rejections.append(
                    {
                        **diagnostics,
                        "reason_code": _compile_reason(inspection),
                    }
                )
        facts: dict[str, JsonValue] = {
            "input_candidate_count": len(value.candidates),
            "survivor_candidate_count": len(survivors),
            "survivors": survivor_records,
            "rejections": rejections,
        }
        if not survivors:
            raise StepFailedError(
                "no candidate compiled",
                failure_code=PreprocessingFailureCode.NO_COMPILABLE_CANDIDATE,
                facts=facts,
            )
        return StepOutput(
            value=IdentifiedCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


def _compile_reason(inspection: CandidateInspection) -> str:
    if inspection.parser_stack_overflow:
        return "parser_stack_overflow"
    if inspection.parser_recursion_overflow:
        return "parser_recursion_overflow"
    return "not_compilable"


__all__ = ["FilterCompilable"]
