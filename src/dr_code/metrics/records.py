"""Self-describing metric records and dataframe-row flattening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Self, TypeAlias

from pydantic import Field, SerializeAsAny, model_validator

from dr_code.metrics.definition import MetricsDefinition
from dr_code.metrics.names import MetricName
from dr_code.models import FrozenModel, settings_payload
from dr_code.trace import TraceProducer


class RecordStatus(StrEnum):
    """The three possible answer shapes for a metric question."""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    OPERATOR_FAILURE = "operator_failure"


MetricScalar: TypeAlias = float | int | str | bool | None


class MetricRecord(FrozenModel):
    """One comparable answer to one declared metric question."""

    metric: MetricName
    metric_version: str
    settings: SerializeAsAny[FrozenModel]

    on_key: str
    producer: TraceProducer
    metrics_definition: MetricsDefinition

    status: RecordStatus
    values: dict[str, MetricScalar] = Field(default_factory=dict)
    absence_failed_step: str | None = None
    absence_cause: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

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
        from dr_code.metrics.registry import REGISTRY

        metric = MetricName(data["metric"])
        data["settings"] = REGISTRY[metric.value].Settings.model_validate(
            settings_payload(data.get("settings", {}))
        )
        return data

    @model_validator(mode="after")
    def validate_answers_a_declared_question(self) -> Self:
        """The record's identity must name a question it nests.

        Scoped to internal consistency: the nested definition is the record's
        own lineage, so ``(metric, on_key, settings)`` must match one of its
        questions exactly as the engine derives them. Deliberately says
        nothing about the live registry — records serialized under older
        operator versions stay loadable.
        """

        for question in self.metrics_definition.questions:
            if (
                question.metric is self.metric
                and question.on == self.on_key
                and question.settings == self.settings
            ):
                return self
        raise ValueError(
            f"record for metric {self.metric} on {self.on_key!r} does not "
            "match any question in its metrics_definition"
        )

    @model_validator(mode="after")
    def validate_answer_shape(self) -> Self:
        has_absence = (
            self.absence_failed_step is not None
            or self.absence_cause is not None
        )
        has_failure = (
            self.failure_type is not None or self.failure_message is not None
        )

        if self.status is RecordStatus.MEASURED:
            if not self.values:
                raise ValueError("measured records require values")
            if has_absence or has_failure:
                raise ValueError(
                    "measured records cannot carry absence or failure fields"
                )
            return self

        if self.values:
            raise ValueError("non-measured records cannot carry values")

        if self.status is RecordStatus.NOT_APPLICABLE:
            if self.absence_failed_step is None or self.absence_cause is None:
                raise ValueError(
                    "not-applicable records require an absence cause"
                )
            if has_failure:
                raise ValueError(
                    "not-applicable records cannot carry failure fields"
                )
            return self

        if self.failure_type is None or self.failure_message is None:
            raise ValueError(
                "operator-failure records require failure type and message"
            )
        if has_absence:
            raise ValueError(
                "operator-failure records cannot carry absence fields"
            )
        return self


def record_rows(
    records: Sequence[MetricRecord],
) -> list[dict[str, object]]:
    """Flatten records into sparse rows with collision-free value columns."""

    rows: list[dict[str, object]] = []
    for record in records:
        row = record.model_dump(mode="python")
        values = row.pop("values")
        assert isinstance(values, dict)
        prefix = str(record.metric)
        row.update({f"{prefix}.{key}": value for key, value in values.items()})
        rows.append(row)
    return rows
