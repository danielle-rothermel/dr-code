"""Materialize the complete ordered inspected-candidate set as output."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.preprocessing.steps.inspected_filter import (
    inspected_candidate_set,
)
from dr_code.trace import Artifact, ArtifactKind


class MaterializeCandidateSet(Step[StepSettings]):
    """Fix the surviving candidates as the definition's complete output.

    Every candidate that survived filtering is returned, in order; nothing
    is selected and nothing is dropped. Which surviving candidate a
    consumer accepts is that consumer's policy, made against the full set
    rather than delegated to a preprocessing step that guesses.

    **``candidate_ordinal`` is defined here.** A candidate's ordinal is its
    zero-based index into the set this step materializes — that is, into
    the set as it stands *after* exact-source deduplication and *after*
    every filter. Ordinals therefore do not index the extracted set, any
    intermediate set, or positions before a duplicate was merged away. A
    coordinate naming ``candidate_ordinal`` n refers to the n-th element of
    this output and to no other set.

    An empty surviving set is the definition's failure: no candidate the
    response contained survived the structural filters.
    """

    NAME: ClassVar[StepName] = StepName.MATERIALIZE_CANDIDATE_SET
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.INSPECTED_CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        materialized = inspected_candidate_set(value)
        if not materialized.candidates:
            raise StepFailedError(
                PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING,
                "no candidate survived filtering",
                evidence={"candidate_count": 0},
            )
        return StepOutput(
            value=materialized,
            facts={"candidate_count": len(materialized.candidates)},
        )


__all__ = ["MaterializeCandidateSet"]
