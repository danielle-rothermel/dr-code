"""Registry-free semantic coordinates for declared metric questions.

A ``MetricsDefinition`` resolves each question's settings against the live
operator registry, which is what makes a declaration executable. A persisted
record cannot depend on that: it is read back long after the registry has
moved on, so it carries a projection instead. That buys independence from
settings churn and from operator implementation and version churn, not from
metric-name churn — ``MetricQuestionCoordinate.metric`` is the closed
``MetricName`` enum, so a coordinate naming a deleted metric does not
validate.

``coordinate_settings`` (``dr_code.trace.provenance``) already solves exactly
this problem for preprocessing component settings, and these coordinates
reuse both its entry model and its value bounds: settings persist as ordered
``ComponentSetting`` entries whose values are bounded scalars or ordered
strings. Structural validation of a coordinate therefore needs no registry
lookup at all.

Operator settings differ from step settings in one way — an operator may
group its parameters into a nested settings model, as ``compressed_length``
groups a compression method and level. ``question_settings`` walks those
groups and names each leaf by its dotted path, so the persisted entry list
stays flat and bounded regardless of how a settings model is composed.
"""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_code.base import FrozenModel
from dr_code.metrics.definition import MetricsDefinition, MetricQuestion
from dr_code.metrics.names import MetricName
from dr_code.metrics.settings import OperatorSettings
from dr_code.trace import ComponentSetting, coordinate_settings

_SETTING_PATH_SEPARATOR = "."


def question_settings(
    settings: OperatorSettings,
) -> tuple[ComponentSetting, ...]:
    """Project operator settings into the bounded persisted entry list.

    Nested settings groups are named by their dotted path. Every leaf goes
    through ``coordinate_settings``, so the persisted value bounds are
    exactly the ones the trace layer enforces — a settings model cannot
    widen the persisted coordinate by nesting.
    """

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
    """Bound one leaf value by projecting it through the trace check."""

    (entry,) = coordinate_settings(_LeafSetting(value=value))
    return ComponentSetting(name=name, value=entry.value)


class _LeafSetting(FrozenModel):
    """One leaf value, wrapped so the trace value bounds apply to it."""

    value: object


class MetricQuestionCoordinate(FrozenModel):
    """Complete semantic coordinate for one declared metric question."""

    metric: MetricName
    on_key: str
    settings: tuple[ComponentSetting, ...] = ()

    @classmethod
    def of(cls, question: MetricQuestion) -> Self:
        """Project a declared question into its persisted coordinate."""

        return cls(
            metric=question.metric,
            on_key=question.on,
            settings=question_settings(question.settings),
        )


class MetricsDefinitionCoordinate(FrozenModel):
    """Complete semantic coordinate for a metrics definition."""

    definition_id: str
    version: str
    questions: tuple[MetricQuestionCoordinate, ...]

    @classmethod
    def of(cls, definition: MetricsDefinition) -> Self:
        """Project a definition into its persisted coordinate."""

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
    "MetricsDefinitionCoordinate",
]
