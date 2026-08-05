"""Python parse outcome metric."""

from __future__ import annotations

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
from dr_code.trace import Artifact, ArtifactKind


class ParseOutcomeResult(OperatorResult):
    UNITS = {
        "parse_ok": MetricFactUnit.BOOLEAN,
        "parse_error": MetricFactUnit.TEXT,
    }

    parse_ok: bool
    parse_error: str | None


class ParseOutcome(MetricOperator[OperatorSettings]):
    NAME = MetricName.PARSE_OUTCOME
    VERSION = "0"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> ParseOutcomeResult:
        _ = aux
        source = artifact_text(value)
        module = ctx.views.parsed_module(source)
        return ParseOutcomeResult(
            parse_ok=module is not None,
            parse_error=ctx.views.parse_error(source),
        )
