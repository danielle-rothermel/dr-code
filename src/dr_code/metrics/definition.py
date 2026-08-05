"""Metric declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import Field, SerializeAsAny, model_validator

from dr_code.metrics.names import MetricName
from dr_code.metrics.settings import OperatorSettings
from dr_code.base import FrozenModel, settings_payload


class MetricQuestion(FrozenModel):
    """One metric family applied to one key in a trace namespace."""

    metric: MetricName
    on: str
    settings: SerializeAsAny[OperatorSettings] = Field(
        default_factory=OperatorSettings
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_settings_model(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "metric" not in data:
            # Let pydantic report the missing discriminator as a
            # ValidationError instead of raising KeyError out of band.
            return data
        metric = MetricName(data["metric"])
        from dr_code.metrics.registry import REGISTRY

        settings_model = REGISTRY[metric.value].Settings
        data["settings"] = settings_model.model_validate(
            settings_payload(data.get("settings", {}))
        )
        return data


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
                question.settings.model_dump_json(),
            )
            if identity in identities:
                raise ValueError(
                    "metric questions must have unique "
                    "(metric, on, settings) triples"
                )
            identities.add(identity)
        return self
