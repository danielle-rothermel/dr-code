"""Derived evaluation values.

A ``Score`` is what an aggregation produced: one number standing for a whole
evaluation. It is strictly derived — it records which facts it came from and
never travels back into them. Metric facts stay observations of one
response; a verdict over many responses is a different kind of thing and
lives here.

Scores reuse ``MetricFactUnit`` rather than defining a parallel vocabulary,
so a score's dimensionality is comparable with the facts it reduced. The one
member a score rejects is ``TEXT``: free-form text is an observation, and a
score is a measurement.
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.aggregation import FactCoordinate
from dr_code.evaluation.coordinates import (
    RepeatPlanCoordinate,
    TaskSetCoordinate,
)
from dr_code.metrics import MetricFactUnit


class EvaluationCoordinate(FrozenModel):
    """The evaluation a score summarizes.

    Nests the plan's identity together with the task set and repeat plan it
    ran over, so a score read back in isolation names its own scope without
    re-reading the plan.
    """

    plan_id: str
    version: str
    task_set: TaskSetCoordinate
    repeat_plan: RepeatPlanCoordinate


class Score(FrozenModel):
    """One finite, united number derived from named source facts.

    ``sources`` is the complete set of fact coordinates the score was
    computed from — question coordinate plus fact name — recorded so the
    derivation is auditable without the aggregation input in hand.
    """

    name: str
    value: float
    unit: MetricFactUnit
    evaluation: EvaluationCoordinate
    sources: tuple[FactCoordinate, ...]

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError(f"score {self.name!r} must be a finite value")
        return self

    @model_validator(mode="after")
    def reject_text_unit(self) -> Self:
        """A score is a measurement, so it is never free-form text."""

        if self.unit is MetricFactUnit.TEXT:
            raise ValueError(
                f"score {self.name!r} cannot have unit "
                f"{MetricFactUnit.TEXT.value!r}: a score is a measurement, "
                "not an observation"
            )
        return self

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if not self.sources:
            raise ValueError(
                f"score {self.name!r} must name the facts it derives from"
            )
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(
                f"score {self.name!r} source facts must be unique"
            )
        return self


__all__ = [
    "EvaluationCoordinate",
    "Score",
]
