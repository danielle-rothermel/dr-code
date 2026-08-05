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
    plan_id: str
    version: str
    task_set: TaskSetCoordinate
    repeat_plan: RepeatPlanCoordinate


class Score(FrozenModel):
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
