"""Retain original candidates and append the legacy last-return salvage."""

from __future__ import annotations

import io
import token
import tokenize
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class _SalvagedPrefix:
    source: str
    end_line: int
    end_column: int


class ExpandLastReturnSalvage(Step):
    """Append changed legacy truncations without replacing original source."""

    NAME: ClassVar[StepName] = StepName.EXPAND_LAST_RETURN_SALVAGE
    VERSION: ClassVar[str] = "4"
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

        for index, source in enumerate(value.candidates):
            salvaged = _salvage_after_last_return(source)
            if salvaged is None or salvaged.source == source:
                continue
            originals.append(salvaged.source)
            if value.lineage:
                operation = ExtractionOperation(
                    kind="drop_after_last_return_salvage",
                    details={
                        "end_line": salvaged.end_line,
                        "end_column": salvaged.end_column,
                    },
                )
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


def _salvage_after_last_return(source: str) -> _SalvagedPrefix | None:
    """Return source through its last complete logical return statement.

    Tokenization distinguishes real keywords from strings and comments, and a
    ``NEWLINE`` token occurs only after bracket continuations close. A later
    malformed token may itself be the trailing junk being removed, but no
    boundary is trusted unless its own logical statement was completed first.
    """
    pending_return = False
    at_statement_start = True
    indent_level = 0
    boundary: tuple[int, int] | None = None
    try:
        for item in tokenize.generate_tokens(io.StringIO(source).readline):
            if item.type == token.ERRORTOKEN and not item.string.isspace():
                break
            if item.type == token.INDENT:
                indent_level += 1
                continue
            if item.type == token.DEDENT:
                indent_level = max(0, indent_level - 1)
                continue
            if item.type in {token.COMMENT, token.NL}:
                continue
            if item.type == token.NEWLINE:
                if pending_return:
                    boundary = item.end
                pending_return = False
                at_statement_start = True
                continue
            if item.type == token.ENDMARKER:
                continue
            if (
                at_statement_start
                and indent_level > 0
                and item.type == token.NAME
                and item.string == "return"
            ):
                pending_return = True
            at_statement_start = False
    except (IndentationError, tokenize.TokenError):
        pass
    if boundary is None or pending_return:
        return None
    end_line, end_column = boundary
    return _SalvagedPrefix(
        source=source[: _source_offset(source, boundary)],
        end_line=end_line,
        end_column=end_column,
    )


def _source_offset(source: str, position: tuple[int, int]) -> int:
    row, column = position
    lines = source.splitlines(keepends=True)
    if row > len(lines):
        return len(source)
    return sum(len(line) for line in lines[: row - 1]) + column


__all__ = ["ExpandLastReturnSalvage"]
