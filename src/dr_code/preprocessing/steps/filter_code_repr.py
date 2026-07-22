"""Drop identified candidates that are code-representation assignments."""

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
    IdentifiedCandidateSetArtifact,
)


CODE_REPR_VARIABLE_NAME = "code"


class FilterCodeRepr(Step):
    """Drop candidates shaped like ``code = \"...\"``."""

    NAME: ClassVar[StepName] = StepName.FILTER_CODE_REPR
    VERSION: ClassVar[str] = "4"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, IdentifiedCandidateSetArtifact)
        survivors = []
        survivor_records: list[dict[str, JsonValue]] = []
        rejections: list[dict[str, JsonValue]] = []
        for index, candidate in enumerate(value.candidates):
            record = _record(index, candidate.lineage.candidate_id)
            if candidate.inspection.is_code_repr_assignment:
                rejections.append(
                    {
                        **record,
                        "reason_code": "code_repr_assignment",
                        **_diagnostics(candidate.inspection),
                    }
                )
            else:
                survivors.append(candidate)
                survivor_records.append(record)
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
            value=IdentifiedCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


__all__ = ["FilterCodeRepr"]
