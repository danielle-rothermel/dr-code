"""Remove candidates made blank by the candidate-cleaning sequence."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import Artifact, ArtifactKind, CodeCandidateSetArtifact


class FilterNonblankCandidates(Step):
    """Keep only nonblank cleaned candidates while retaining their lineage."""

    NAME: ClassVar[StepName] = StepName.FILTER_NONBLANK_CANDIDATES
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        candidates: list[str] = []
        lineage = []
        rejections: list[dict[str, int | str]] = []
        for index, candidate in enumerate(value.candidates):
            if candidate.strip():
                candidates.append(candidate)
                if value.lineage:
                    lineage.append(value.lineage_at(index))
            else:
                rejections.append(
                    {"index": index, "reason": "blank_or_whitespace"}
                )

        facts = {
            "input_candidate_count": len(value.candidates),
            "output_candidate_count": len(candidates),
            "rejections": rejections,
        }
        if not candidates:
            raise StepFailedError(
                "no nonblank cleaned candidate",
                failure_code=(
                    PreprocessingFailureCode.NO_NONBLANK_CLEANED_CANDIDATE
                ),
                facts=facts,
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(candidates), lineage=tuple(lineage)
            ),
            facts=facts,
        )


__all__ = ["FilterNonblankCandidates"]
