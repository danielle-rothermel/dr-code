"""Retain original candidates and append the legacy last-return salvage."""

from __future__ import annotations

import io
import token
import tokenize
from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
    ExtractionOperation,
)


class ExpandLastReturnSalvage(Step):
    """Append changed legacy truncations without replacing original source."""

    NAME: ClassVar[StepName] = StepName.EXPAND_LAST_RETURN_SALVAGE
    VERSION: ClassVar[str] = "2"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, CodeCandidateSetArtifact)
        originals = list(value.candidates)
        lineage = (
            [
                value.lineage_at(index).model_copy(
                    update={"candidate_id": None}
                )
                for index in range(len(value.candidates))
            ]
            if value.lineage
            else []
        )
        repairs: list[dict[str, int]] = []
        operation = ExtractionOperation(kind="drop_after_last_return_salvage")

        for index, source in enumerate(value.candidates):
            salvaged = _salvage_after_last_return(source)
            if salvaged is None or salvaged == source:
                continue
            originals.append(salvaged)
            if value.lineage:
                lineage.append(
                    _append_operation(value.lineage_at(index), operation)
                )
            repairs.append(
                {
                    "input_index": index,
                    "output_index": len(originals) - 1,
                }
            )

        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(originals), lineage=tuple(lineage)
            ),
            facts={
                "input_candidate_count": len(value.candidates),
                "output_candidate_count": len(originals),
                "salvage_candidate_count": len(repairs),
                "repairs": repairs,
            },
        )


def _append_operation(
    lineage: CandidateLineage, operation: ExtractionOperation
) -> CandidateLineage:
    origins = lineage.origins or (CandidateOrigin(path=()),)
    return CandidateLineage(
        origins=tuple(
            CandidateOrigin(path=(*origin.path, operation))
            for origin in origins
        )
    )


def _salvage_after_last_return(source: str) -> str | None:
    """Return source through its last complete logical return statement.

    Tokenization distinguishes real keywords from strings and comments, and a
    ``NEWLINE`` token occurs only after bracket continuations close. A later
    malformed token may itself be the trailing junk being removed, but no
    boundary is trusted unless its own logical statement was completed first.
    """
    pending_return = False
    boundary: tuple[int, int] | None = None
    try:
        for item in tokenize.generate_tokens(io.StringIO(source).readline):
            if item.type == token.ERRORTOKEN and not item.string.isspace():
                break
            if item.type == token.NAME and item.string == "return":
                pending_return = True
            elif pending_return and item.type == token.NEWLINE:
                boundary = item.end
                pending_return = False
    except (IndentationError, tokenize.TokenError):
        pass
    if boundary is None or pending_return:
        return None
    return source[: _source_offset(source, boundary)]


def _source_offset(source: str, position: tuple[int, int]) -> int:
    row, column = position
    lines = source.splitlines(keepends=True)
    if row > len(lines):
        return len(source)
    return sum(len(line) for line in lines[: row - 1]) + column


__all__ = ["ExpandLastReturnSalvage"]
