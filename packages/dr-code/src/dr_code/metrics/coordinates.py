from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel
from dr_code.metrics.definition import MetricsDefinition, MetricQuestion
from dr_code.metrics.names import MetricName
from dr_code.metrics.settings import OperatorSettings
from dr_code.trace import ComponentSetting, coordinate_settings

_SETTING_PATH_SEPARATOR = "."


def question_settings(
    settings: OperatorSettings,
) -> tuple[ComponentSetting, ...]:
    entries: list[ComponentSetting] = []
    for name, value in dict(settings).items():
        if isinstance(value, FrozenModel):
            entries.extend(
                ComponentSetting(
                    name=f"{name}{_SETTING_PATH_SEPARATOR}{nested.name}",
                    value=nested.value,
                )
                for nested in coordinate_settings(value)
            )
            continue
        entries.append(_leaf_entry(name, value))
    return tuple(entries)


def _leaf_entry(name: str, value: object) -> ComponentSetting:
    (entry,) = coordinate_settings(_LeafSetting(value=value))
    return ComponentSetting(name=name, value=entry.value)


class _LeafSetting(FrozenModel):
    value: object


class MetricQuestionCoordinate(FrozenModel):
    metric: MetricName
    on_key: str
    settings: tuple[ComponentSetting, ...] = ()

    @classmethod
    def of(cls, question: MetricQuestion) -> Self:
        return cls(
            metric=question.metric,
            on_key=question.on,
            settings=question_settings(question.settings),
        )


class MetricValueCoordinate(FrozenModel):
    question: MetricQuestionCoordinate
    value: str

    @model_validator(mode="after")
    def validate_value_name(self) -> Self:
        if not self.value:
            raise ValueError("a metric value coordinate must name a value")
        if "." in self.value:
            raise ValueError(
                f"metric value name {self.value!r} must not contain '.': "
                "no metric value can carry a dotted name"
            )
        return self


class MetricsDefinitionCoordinate(FrozenModel):
    definition_id: str
    version: str
    questions: tuple[MetricQuestionCoordinate, ...]

    @classmethod
    def of(cls, definition: MetricsDefinition) -> Self:
        return cls(
            definition_id=definition.definition_id,
            version=definition.version,
            questions=tuple(
                MetricQuestionCoordinate.of(question)
                for question in definition.questions
            ),
        )

    @model_validator(mode="after")
    def reject_duplicate_questions(self) -> Self:
        if len(set(self.questions)) != len(self.questions):
            raise ValueError(
                "metric question coordinates must be unique within a "
                "metrics definition coordinate"
            )
        return self


__all__ = [
    "MetricQuestionCoordinate",
    "MetricValueCoordinate",
    "MetricsDefinitionCoordinate",
]
