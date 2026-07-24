"""Unit-carrying Metric Facts, Scores, and operator lineage.

A ``MetricRecord`` (``dr_code.metrics.records``) is the persisted answer to one
declared metric question. A ``MetricFact`` is one named value inside such an
answer, projected with the explicit unit its operator declares and stamped with
the coordinates that produced it. A ``Score`` is a deterministic derivation
from named facts.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from dr_code.eval.lifecycle import ConfigCoordinate
from dr_code.metrics.names import MetricName
from dr_code.models import FrozenModel

StrictFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]

FactScalar: TypeAlias = StrictFiniteFloat | StrictInt | StrictStr | StrictBool

FactValue: TypeAlias = FactScalar | None
SCORE_SCHEMA_VERSION: Final = 1


def _validate_fact_scalar(value: object) -> object:
    if type(value) not in (float, int, str, bool):
        raise ValueError("fact values must be strict JSON scalars")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("fact values must be finite")
    return value


class Applicability(StrEnum):
    """Whether one declared fact has a value for one observation."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class AbsenceMode(StrEnum):
    """Why an observation carries no measurable input."""

    PREPROCESSING_FAILURE = "preprocessing_failure"
    NO_INPUT = "no_input"
    NO_TRACE = "no_trace"
    MISSING_TRACE_KEY = "missing_trace_key"
    EMPTY_CANDIDATE_SET = "empty_candidate_set"


class OperatorLineage(FrozenModel):
    """The resolved coordinates that produced one fact."""

    evaluation_procedure_config: ConfigCoordinate
    operator: MetricName
    operator_version: str
    on_key: str
    step: str | None = None
    step_version: str | None = None

    @model_validator(mode="after")
    def validate_step_pair(self) -> Self:
        if not self.operator_version:
            raise ValueError("operator lineage requires an operator version")
        if (self.step is None) != (self.step_version is None):
            raise ValueError("step and step_version must be set together")
        return self


class MetricFact(FrozenModel):
    """One named measured value with unit, applicability, and lineage."""

    name: str
    value: FactValue
    unit: str
    applicability: Applicability
    reason: str | None = None
    lineage: OperatorLineage

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> object:
        return None if value is None else _validate_fact_scalar(value)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.name:
            raise ValueError("a metric fact requires a name")
        if not self.unit:
            raise ValueError("a metric fact requires an explicit unit")
        if self.applicability is Applicability.APPLICABLE:
            if self.value is None:
                raise ValueError("applicable metric facts require a value")
            if self.reason is not None:
                raise ValueError(
                    "applicable metric facts cannot carry an absence reason"
                )
        else:
            if self.value is not None:
                raise ValueError(
                    "not-applicable metric facts cannot carry a value"
                )
            if not self.reason:
                raise ValueError(
                    "not-applicable metric facts require an explicit reason"
                )
        return self


class Score(FrozenModel):
    """A deterministic derivation from named Metric Facts."""

    schema_version: Literal[1]
    name: str
    value: FactScalar
    unit: str
    evaluation_procedure_config: ConfigCoordinate
    derived_from: tuple[str, ...]

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> object:
        return _validate_fact_scalar(value)

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if not self.unit:
            raise ValueError("a score requires an explicit unit")
        if not self.derived_from:
            raise ValueError("a score requires source fact names")
        return self


__all__ = [
    "AbsenceMode",
    "Applicability",
    "FactScalar",
    "FactValue",
    "MetricFact",
    "OperatorLineage",
    "SCORE_SCHEMA_VERSION",
    "Score",
]
