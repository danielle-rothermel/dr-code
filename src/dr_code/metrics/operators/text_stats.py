"""Character, byte, line, and word statistics."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping

from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    artifact_text,
)
from dr_code.metrics.records import MetricScalar
from dr_code.trace import Artifact, ArtifactKind

_TEXT_ENCODING = "utf-8"
_WORD_RE = re.compile(r"\b\w+\b")
_OPERATOR_CHARS = frozenset("+-*/%=<>!&|^~:@")


class TextStats(MetricOperator):
    NAME = MetricName.TEXT_STATS
    VERSION = "1"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> dict[str, MetricScalar]:
        _ = aux, ctx
        text = artifact_text(value)
        words = _WORD_RE.findall(text)
        word_lengths = [len(word) for word in words]
        return {
            "character_count": len(text),
            "byte_count": len(text.encode(_TEXT_ENCODING)),
            "line_count": len(text.split("\n")) if text else 0,
            "nonempty_line_count": sum(
                1 for line in text.splitlines() if line.strip()
            ),
            "word_count": len(words),
            "average_word_length": (
                sum(word_lengths) / len(word_lengths) if word_lengths else None
            ),
            "punctuation_count": sum(
                1 for character in text if character in string.punctuation
            ),
            "symbol_count": sum(
                1 for character in text if character in _OPERATOR_CHARS
            ),
        }
