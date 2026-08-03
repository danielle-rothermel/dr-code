"""Character, byte, line, and word statistics."""

from __future__ import annotations

import string
from collections.abc import Mapping

from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
    OperatorSettings,
    artifact_text,
)
from dr_code.text_analysis import OPERATOR_CHARS, WORD_RE
from dr_code.trace import Artifact, ArtifactKind

_TEXT_ENCODING = "utf-8"


class TextStatsResult(OperatorResult):
    character_count: int
    byte_count: int
    line_count: int
    nonempty_line_count: int
    word_count: int
    average_word_length: float | None
    punctuation_count: int
    symbol_count: int


class TextStats(MetricOperator[OperatorSettings]):
    NAME = MetricName.TEXT_STATS
    VERSION = "0"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> TextStatsResult:
        _ = aux, ctx
        text = artifact_text(value)
        words = WORD_RE.findall(text)
        word_lengths = [len(word) for word in words]
        return TextStatsResult(
            character_count=len(text),
            byte_count=len(text.encode(_TEXT_ENCODING)),
            line_count=len(text.split("\n")) if text else 0,
            nonempty_line_count=sum(
                1 for line in text.splitlines() if line.strip()
            ),
            word_count=len(words),
            average_word_length=(
                sum(word_lengths) / len(word_lengths) if word_lengths else None
            ),
            punctuation_count=sum(
                1 for character in text if character in string.punctuation
            ),
            symbol_count=sum(
                1 for character in text if character in OPERATOR_CHARS
            ),
        )
