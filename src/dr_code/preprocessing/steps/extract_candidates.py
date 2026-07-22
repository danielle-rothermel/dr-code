"""Thin preprocessing adapter for modular candidate extraction."""

from __future__ import annotations

from collections import Counter
from typing import ClassVar

from dr_code.preprocessing.extraction import extract_candidate_drafts
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateLineage,
    CodeCandidateSetArtifact,
    TextArtifact,
)


class ExtractCandidates(Step[StepSettings]):
    """Emit every candidate draft with its complete ordered origin path."""

    NAME: ClassVar[StepName] = StepName.EXTRACT_CANDIDATES
    VERSION: ClassVar[str] = "3"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        drafts = extract_candidate_drafts(value.text)
        operation_counts = Counter(
            operation.kind
            for draft in drafts
            for operation in draft.origin.path
        )
        facts = {
            "candidate_count": len(drafts),
            "operation_counts": [
                {"kind": kind, "count": count}
                for kind, count in sorted(operation_counts.items())
            ],
            "paths": [
                draft.origin.model_dump(mode="json") for draft in drafts
            ],
        }
        if not drafts:
            raise StepFailedError(
                "no code candidates extracted",
                failure_code=PreprocessingFailureCode.NO_CODE_CANDIDATES,
                facts=facts,
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(draft.source for draft in drafts),
                lineage=tuple(
                    CandidateLineage(origins=(draft.origin,))
                    for draft in drafts
                ),
            ),
            facts=facts,
        )


__all__ = ("ExtractCandidates",)
