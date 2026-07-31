"""Keep identified candidates with a function at module scope."""

from __future__ import annotations

from typing import ClassVar

from pydantic import JsonValue

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.preprocessing.steps.filter_compilable import _compile_reason
from dr_code.preprocessing.steps.filter_plain_literal import (
    _diagnostics,
    _record,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    IdentifiedCandidateSetArtifact,
)


class FilterHasTopLevelFunction(Step):
    """Keep candidates with at least one top-level sync or async function."""

    NAME: ClassVar[StepName] = StepName.FILTER_HAS_TOP_LEVEL_FUNCTION
    VERSION: ClassVar[str] = "3"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, IdentifiedCandidateSetArtifact)
        survivors = []
        survivor_details: list[dict[str, JsonValue]] = []
        rejections: list[dict[str, JsonValue]] = []
        for index, candidate in enumerate(value.candidates):
            inspection = candidate.inspection
            diagnostics = {
                **_record(index, candidate.lineage.candidate_id),
                **_diagnostics(inspection),
            }
            if not inspection.compile_ok:
                rejections.append(
                    {
                        **diagnostics,
                        "reason_code": _compile_reason(inspection),
                    }
                )
                continue
            names = list(inspection.top_level_function_names)
            async_names = list(inspection.top_level_async_function_names)
            details: dict[str, JsonValue] = {
                **diagnostics,
                "top_level_function_count": len(names),
                "top_level_function_names": names,
                "top_level_async_function_names": async_names,
                "has_async_top_level_function": bool(async_names),
            }
            if not names:
                rejections.append(
                    {**details, "reason_code": "no_top_level_function"}
                )
                continue
            survivors.append(candidate)
            survivor_details.append(details)
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
            value=IdentifiedCandidateSetArtifact(candidates=tuple(survivors)),
            facts=facts,
        )


__all__ = ["FilterHasTopLevelFunction"]
