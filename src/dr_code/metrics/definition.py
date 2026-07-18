"""Metric declarations and deterministic definition identity."""

from __future__ import annotations

import json
from typing import Self

from pydantic import JsonValue, model_validator

from dr_code.metrics.names import MetricName
from dr_code.models import FrozenModel
from dr_code.trace import stable_hash


class MetricQuestion(FrozenModel):
    """One metric family applied to one key in a trace namespace."""

    metric: MetricName
    on: str
    settings: dict[str, JsonValue] = {}


class MetricsDefinition(FrozenModel):
    """An ordered, versioned collection of metric questions."""

    definition_id: str
    version: str
    questions: tuple[MetricQuestion, ...]

    @model_validator(mode="after")
    def reject_duplicate_questions(self) -> Self:
        identities: set[tuple[MetricName, str, str]] = set()
        for question in self.questions:
            identity = (
                question.metric,
                question.on,
                json.dumps(question.settings, sort_keys=True),
            )
            if identity in identities:
                raise ValueError(
                    "metric questions must have unique "
                    "(metric, on, settings) triples"
                )
            identities.add(identity)
        return self


def metrics_definition_hash(definition: MetricsDefinition) -> str:
    """Return the canonical BLAKE2b identity of ``definition``."""

    return stable_hash(definition)
