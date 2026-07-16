"""Python parse outcome metric."""

from __future__ import annotations

from collections.abc import Mapping

from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    artifact_text,
)
from dr_code.metrics.records import MetricScalar
from dr_code.trace import Artifact, ArtifactKind


class ParseOutcome(MetricOperator):
    NAME = MetricName.PARSE_OUTCOME
    VERSION = "1"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> dict[str, MetricScalar]:
        _ = aux
        source = artifact_text(value)
        module = ctx.views.parsed_module(source)
        return {
            "parse_ok": module is not None,
            "parse_error": ctx.views.parse_error(source),
        }
