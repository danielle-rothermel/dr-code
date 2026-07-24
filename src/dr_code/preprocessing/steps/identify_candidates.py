"""Identify and inspect each unique cleaned candidate source once."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.identification import identify_candidates
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import Artifact, ArtifactKind, CodeCandidateSetArtifact


class IdentifyCandidates(Step):
    """Canonicalize and inspect candidates for downstream policy filters."""

    NAME: ClassVar[StepName] = StepName.IDENTIFY_CANDIDATES
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.IDENTIFIED_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        if not value.candidates:
            raise StepFailedError(
                "no candidates available for identification",
                failure_code=(
                    PreprocessingFailureCode.NO_CANDIDATES_TO_IDENTIFY
                ),
                facts={"input_candidate_count": 0},
            )
        identified, facts = identify_candidates(value)
        return StepOutput(value=identified, facts=facts)


__all__ = ["IdentifyCandidates"]
