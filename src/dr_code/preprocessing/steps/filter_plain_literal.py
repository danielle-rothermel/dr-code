"""Drop identified candidates that are plain literal modules."""

from __future__ import annotations

from typing import ClassVar

from pydantic import JsonValue

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateInspection,
    IdentifiedCandidateSetArtifact,
)


class FilterPlainLiteral(Step):
    """Drop candidates that are a single container-literal expression."""

    NAME: ClassVar[StepName] = StepName.FILTER_PLAIN_LITERAL
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
            if candidate.inspection.is_plain_literal_module:
                rejections.append(
                    {
                        **record,
                        "reason_code": "plain_literal_module",
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
                "no candidate survived plain-literal filtering",
                failure_code=PreprocessingFailureCode.PLAIN_LITERAL_ONLY,
                facts=facts,
            )
        return StepOutput(
            value=IdentifiedCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


def _record(index: int, candidate_id: str | None) -> dict[str, JsonValue]:
    record: dict[str, JsonValue] = {"input_index": index}
    if candidate_id is not None:
        record["candidate_id"] = candidate_id
    return record


def _diagnostics(
    inspection: CandidateInspection,
) -> dict[str, JsonValue]:
    return {
        "parse_ok": inspection.parse_ok,
        "parse_error": inspection.parse_error,
        "compile_ok": inspection.compile_ok,
        "compile_error": inspection.compile_error,
        "compile_warnings": list(inspection.compile_warnings),
    }


__all__ = ["FilterPlainLiteral"]
