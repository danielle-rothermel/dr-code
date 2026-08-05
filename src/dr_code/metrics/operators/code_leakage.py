from __future__ import annotations

import keyword
import string
from collections.abc import Mapping

from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
    artifact_text,
)
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.units import MetricFactUnit
from dr_code.core.source.text_analysis import (
    CODE_LIKE_LINE_RE,
    FENCE_LINE_RE,
    OPERATOR_CHARS,
    WORD_RE,
)
from dr_code.trace import Artifact, ArtifactKind

_CODE_MARKERS = frozenset({"def", "return", "import", "class"})


class CodeLeakageSettings(OperatorSettings):
    task_names: tuple[str, ...] = ()


class CodeLeakageResult(OperatorResult):
    UNITS = {
        "keyword_count": MetricFactUnit.COUNT,
        "code_marker_count": MetricFactUnit.COUNT,
        "fenced_code_block_count": MetricFactUnit.COUNT,
        "code_like_line_count": MetricFactUnit.LINES,
        "operator_count": MetricFactUnit.COUNT,
        "punctuation_density": MetricFactUnit.RATIO,
        "task_name_hit_count": MetricFactUnit.COUNT,
    }

    keyword_count: int
    code_marker_count: int
    fenced_code_block_count: int
    code_like_line_count: int
    operator_count: int
    punctuation_density: float | None
    task_name_hit_count: int


class CodeLeakage(MetricOperator[CodeLeakageSettings]):
    NAME = MetricName.CODE_LEAKAGE
    VERSION = "0"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})
    Settings = CodeLeakageSettings

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> CodeLeakageResult:
        _ = aux, ctx
        text = artifact_text(value)
        words = WORD_RE.findall(text)
        punctuation_count = sum(
            1 for character in text if character in string.punctuation
        )
        return CodeLeakageResult(
            keyword_count=sum(1 for word in words if keyword.iskeyword(word)),
            code_marker_count=sum(
                1 for word in words if word in _CODE_MARKERS
            ),
            fenced_code_block_count=sum(
                1 for line in text.splitlines() if FENCE_LINE_RE.match(line)
            )
            // 2,
            code_like_line_count=sum(
                1
                for line in text.splitlines()
                if CODE_LIKE_LINE_RE.match(line)
            ),
            operator_count=sum(
                1 for character in text if character in OPERATOR_CHARS
            ),
            punctuation_density=(
                punctuation_count / len(text) if text else None
            ),
            task_name_hit_count=sum(
                text.count(task_name)
                for task_name in self.settings.task_names
                if task_name
            ),
        )
