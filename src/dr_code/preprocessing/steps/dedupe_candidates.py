"""Deduplicate cleaned candidates while merging their extraction origins."""

from __future__ import annotations

import hashlib
from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
)


def _candidate_id(source: str) -> str:
    """Return a content-derived identity stable across processes and runs."""
    digest = hashlib.blake2b(
        source.encode("utf-8"), digest_size=16
    ).hexdigest()
    return f"candidate-{digest}"


def _origins(
    value: CodeCandidateSetArtifact, index: int
) -> tuple[CandidateOrigin, ...]:
    return value.lineage_at(index).origins


class DedupeCandidates(Step):
    """Keep first exact cleaned source and merge every duplicate's origins."""

    NAME: ClassVar[StepName] = StepName.DEDUPE_CANDIDATES
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        if not value.candidates:
            raise StepFailedError(
                "no candidates available for deduplication",
                failure_code=(
                    PreprocessingFailureCode.NO_CANDIDATES_TO_DEDUPE
                ),
                facts={"input": {"candidate_count": 0, "lineage_count": 0}},
            )

        sources: list[str] = []
        kept_indices: list[int] = []
        merged_origins: list[list[CandidateOrigin]] = []
        source_index: dict[str, int] = {}
        duplicate_indexes: dict[int, list[int]] = {}

        for index, source in enumerate(value.candidates):
            kept_position = source_index.get(source)
            if kept_position is None:
                source_index[source] = len(sources)
                sources.append(source)
                kept_indices.append(index)
                merged_origins.append(list(_origins(value, index)))
                continue
            merged_origins[kept_position].extend(_origins(value, index))
            duplicate_indexes.setdefault(kept_position, []).append(index)

        lineage = tuple(
            CandidateLineage(
                candidate_id=_candidate_id(source), origins=tuple(origins)
            )
            for source, origins in zip(sources, merged_origins, strict=True)
        )
        duplicate_groups = [
            {
                "candidate_id": lineage[position].candidate_id or "",
                "first_input_index": kept_indices[position],
                "duplicate_input_indexes": indexes,
                "merged_origins": [
                    origin.model_dump(mode="json")
                    for origin in lineage[position].origins
                ],
            }
            for position, indexes in duplicate_indexes.items()
        ]
        facts = {
            "input": {
                "candidate_count": len(value.candidates),
                "lineage_count": len(value.lineage),
                "lineage": [
                    item.model_dump(mode="json") for item in value.lineage
                ],
            },
            "output": {
                "candidate_count": len(sources),
                "lineage": [item.model_dump(mode="json") for item in lineage],
            },
            "duplicate_groups": duplicate_groups,
        }
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(sources), lineage=lineage
            ),
            facts=facts,
        )


__all__ = ["DedupeCandidates"]
