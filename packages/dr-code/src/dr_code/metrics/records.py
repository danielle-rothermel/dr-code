from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import Field, TypeAdapter, model_validator

from dr_code.core.models import FrozenModel
from dr_code.metrics.coordinates import (
    MetricQuestionCoordinate,
    MetricsDefinitionCoordinate,
)
from dr_code.metrics.units import MetricValueUnit
from dr_code.trace import Absent, TraceProducer

METRIC_RECORD_SCHEMA_VERSION: Final = 1


class RecordStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    OPERATOR_FAILURE = "operator_failure"


MetricScalar: TypeAlias = float | int | str | bool | None


class MetricValue(FrozenModel):
    name: str
    value: MetricScalar
    unit: MetricValueUnit

    @model_validator(mode="after")
    def reject_non_finite_value(self) -> Self:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(
                f"metric value {self.name!r} must be a finite value"
            )
        return self

    @model_validator(mode="after")
    def validate_name(self) -> Self:
        if not self.name:
            raise ValueError("metric value name must not be empty")
        if "." in self.name:
            raise ValueError(
                f"metric value name {self.name!r} must not contain '.': "
                "the dot separates a value column from its unit column"
            )
        return self


class OperatorFailure(FrozenModel):
    failure_type: str
    failure_message: str


class MetricRecordId(FrozenModel):
    question: MetricQuestionCoordinate
    metric_version: str
    producer: TraceProducer
    metrics_definition: MetricsDefinitionCoordinate

    @model_validator(mode="after")
    def validate_answers_a_declared_question(self) -> Self:
        if self.question not in self.metrics_definition.questions:
            raise ValueError(
                f"record for metric {self.question.metric} on "
                f"{self.question.on_key!r} does not match any question in "
                "its metrics_definition"
            )
        return self


class MeasuredRecord(FrozenModel):
    schema_version: Literal[1] = METRIC_RECORD_SCHEMA_VERSION
    status: Literal[RecordStatus.MEASURED] = RecordStatus.MEASURED
    identity: MetricRecordId
    values: tuple[MetricValue, ...]

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.values:
            raise ValueError("measured records require at least one value")
        names = [value.name for value in self.values]
        if len(set(names)) != len(names):
            raise ValueError("measured record value names must be unique")
        return self


class NotApplicableRecord(FrozenModel):
    schema_version: Literal[1] = METRIC_RECORD_SCHEMA_VERSION
    status: Literal[RecordStatus.NOT_APPLICABLE] = RecordStatus.NOT_APPLICABLE
    identity: MetricRecordId
    absence: Absent


class OperatorFailureRecord(FrozenModel):
    schema_version: Literal[1] = METRIC_RECORD_SCHEMA_VERSION
    status: Literal[RecordStatus.OPERATOR_FAILURE] = (
        RecordStatus.OPERATOR_FAILURE
    )
    identity: MetricRecordId
    failure: OperatorFailure


MetricRecord: TypeAlias = Annotated[
    MeasuredRecord | NotApplicableRecord | OperatorFailureRecord,
    Field(discriminator="status"),
]

METRIC_RECORD_ADAPTER: Final = TypeAdapter[MetricRecord](MetricRecord)


def record_rows(
    records: Sequence[MetricRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        row = record.model_dump(mode="python")
        identity = row.pop("identity")
        question = identity.pop("question")
        row["metric"] = question["metric"]
        row["on_key"] = question["on_key"]
        row["question_settings"] = question["settings"]
        row.update(identity)

        values = row.pop("values", ())
        prefix = str(record.identity.question.metric)
        for value in values:
            column = f"{prefix}.{value['name']}"
            row[column] = value["value"]
            row[f"{column}.unit"] = value["unit"]
        rows.append(row)
    return rows


__all__ = [
    "METRIC_RECORD_ADAPTER",
    "METRIC_RECORD_SCHEMA_VERSION",
    "MeasuredRecord",
    "MetricValue",
    "MetricRecord",
    "MetricRecordId",
    "MetricScalar",
    "NotApplicableRecord",
    "OperatorFailure",
    "OperatorFailureRecord",
    "RecordStatus",
]
