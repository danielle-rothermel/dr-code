from __future__ import annotations

import math
from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.coordinates import (
    SamplingPlanCoordinate,
    TaskSetCoordinate,
)
from dr_code.metrics import MetricValueCoordinate, MetricValueUnit


class EvaluationCoordinate(FrozenModel):
    plan_id: str
    version: str
    task_set: TaskSetCoordinate
    sampling_plan: SamplingPlanCoordinate


class Score(FrozenModel):
    name: str
    value: float
    unit: MetricValueUnit
    evaluation: EvaluationCoordinate
    sources: tuple[MetricValueCoordinate, ...]

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError(f"score {self.name!r} must be a finite value")
        return self

    @model_validator(mode="after")
    def reject_text_unit(self) -> Self:
        if self.unit is MetricValueUnit.TEXT:
            raise ValueError(
                f"score {self.name!r} cannot have unit "
                f"{MetricValueUnit.TEXT.value!r}: a score is a measurement, "
                "not an observation"
            )
        return self

    @model_validator(mode="after")
    def validate_sources(self) -> Self:
        if not self.sources:
            raise ValueError(
                f"score {self.name!r} must name the metric values it derives from"
            )
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(
                f"score {self.name!r} source metric values must be unique"
            )
        return self


__all__ = [
    "EvaluationCoordinate",
    "Score",
]
