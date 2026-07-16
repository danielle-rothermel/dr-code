"""Signals that prose contains Python or benchmark-specific content."""

from __future__ import annotations

import keyword
import re
import string
from collections.abc import Mapping

from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorSettings,
    artifact_text,
)
from dr_code.metrics.records import MetricScalar
from dr_code.trace import Artifact, ArtifactKind

_WORD_RE = re.compile(r"\b\w+\b")
_FENCED_CODE_RE = re.compile(r"```|~~~")
_CODE_LIKE_LINE_RE = re.compile(
    r"^\s*(def |async def |class |import |from |return\b|if |for |while |"
    r"try:|except\b|with |[A-Za-z_]\w*\s*=)"
)
_CODE_MARKERS = frozenset({"def", "return", "import", "class"})
_OPERATOR_CHARS = frozenset("+-*/%=<>!&|^~:@")


class CodeLeakageSettings(OperatorSettings):
    task_names: tuple[str, ...] = ()


class CodeLeakage(MetricOperator):
    NAME = MetricName.CODE_LEAKAGE
    VERSION = "1"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})
    Settings = CodeLeakageSettings

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> dict[str, MetricScalar]:
        _ = aux, ctx
        text = artifact_text(value)
        words = _WORD_RE.findall(text)
        punctuation_count = sum(
            1 for character in text if character in string.punctuation
        )
        settings = self.settings
        assert isinstance(settings, CodeLeakageSettings)
        return {
            "keyword_count": sum(
                1 for word in words if keyword.iskeyword(word)
            ),
            "code_marker_count": sum(
                1 for word in words if word in _CODE_MARKERS
            ),
            "fenced_code_block_count": len(
                _FENCED_CODE_RE.findall(text)
            )
            // 2,
            "code_like_line_count": sum(
                1
                for line in text.splitlines()
                if _CODE_LIKE_LINE_RE.match(line)
            ),
            "operator_count": sum(
                1 for character in text if character in _OPERATOR_CHARS
            ),
            "punctuation_density": (
                punctuation_count / len(text) if text else None
            ),
            "task_name_hit_count": sum(
                text.count(task_name)
                for task_name in settings.task_names
                if task_name
            ),
        }
