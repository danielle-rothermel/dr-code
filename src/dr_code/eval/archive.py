"""Durable readers for current and published MetricRecord payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Annotated, Final, Literal, NoReturn, Self, TypeAlias, cast

from pydantic import (
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_code.eval.facts import (
    AbsenceMode,
    Applicability,
    EvaluationProcedureConfigHash,
    MetricFact,
    MetricRecord,
    OperatorCoordinates,
    RecordStatus,
)
from dr_code.eval.identity import (
    SCHEMA_METRIC_QUESTION_BINDING,
    identity_hash_for,
)
from dr_code.eval.immutable_json import freeze_json, thaw_json
from dr_code.models import FrozenModel
from dr_code.trace.archive import ArchivedTraceProducerV2
from dr_code.trace.observation import SampleIdentity
from dr_code.trace.provenance import TraceProducer

LEGACY_UNSPECIFIED_FAILURE_CODE: Final = "legacy_unspecified"
PersistedMetricRecordPayload: TypeAlias = (
    Mapping[str, object] | str | bytes | bytearray
)
_ArchivedStrictFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]
_ArchivedFactScalar: TypeAlias = (
    _ArchivedStrictFiniteFloat | StrictInt | StrictStr | StrictBool
)
_ArchivedFactValue: TypeAlias = _ArchivedFactScalar | None


def _validate_archived_fact_scalar(value: object) -> object:
    if type(value) not in (float, int, str, bool):
        raise ValueError("fact values must be strict JSON scalars")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("fact values must be finite")
    return value


class ArchivedOperatorLineageV1(FrozenModel):
    """Published lineage before operator implementation identity."""

    evaluation_procedure_config_hash: EvaluationProcedureConfigHash
    question_identity_hash: str
    operator: str
    operator_version: str
    step: str | None = None
    step_version: str | None = None

    @model_validator(mode="after")
    def validate_step_pair(self) -> Self:
        if (self.step is None) != (self.step_version is None):
            raise ValueError("step and step_version must be set together")
        return self


class ArchivedOperatorCoordinatesV1(FrozenModel):
    """Published operator coordinates before implementation identity."""

    name: str
    version: str
    settings: tuple[tuple[str, JsonValue], ...]

    @field_validator("settings", mode="after")
    @classmethod
    def freeze_settings(
        cls, value: tuple[tuple[str, JsonValue], ...]
    ) -> tuple[tuple[str, JsonValue], ...]:
        return tuple(
            (name, cast(JsonValue, freeze_json(setting)))
            for name, setting in value
        )

    @field_serializer("settings")
    def serialize_settings(
        self, value: tuple[tuple[str, JsonValue], ...]
    ) -> list[list[JsonValue]]:
        return [[name, thaw_json(setting)] for name, setting in value]

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if not self.name or not self.version:
            raise ValueError("operator name and version must be non-empty")
        names = [name for name, _value in self.settings]
        if len(names) != len(set(names)):
            raise ValueError("operator setting names must be unique")
        if names != sorted(names):
            raise ValueError("operator settings must use canonical key order")
        return self

    def question_identity_hash(self, *, on_key: str) -> str:
        return identity_hash_for(
            schema=SCHEMA_METRIC_QUESTION_BINDING,
            payload={
                "metric": self.name,
                "on": on_key,
                "settings": [
                    [name, thaw_json(value)] for name, value in self.settings
                ],
            },
        )


class ArchivedMetricFactV1(FrozenModel):
    """Published metric fact before operator implementation identity."""

    name: str
    value: _ArchivedFactValue
    unit: str
    applicability: Applicability
    reason: str | None = None
    lineage: ArchivedOperatorLineageV1

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> object:
        return None if value is None else _validate_archived_fact_scalar(value)

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


class ArchivedMetricRecordV1(FrozenModel):
    """Readable MetricRecord schema 1, intentionally not promotable."""

    schema_version: Literal[1]
    question: str
    question_identity_hash: str
    on_key: str
    evaluation_procedure_config_hash: EvaluationProcedureConfigHash
    trace_producer: ArchivedTraceProducerV2
    operator: ArchivedOperatorCoordinatesV1
    status: RecordStatus
    facts: tuple[ArchivedMetricFactV1, ...] = ()
    absence_mode: AbsenceMode | None = None
    absence_cause: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.operator.name != self.question:
            raise ValueError(
                "record operator coordinates must match the question"
            )
        if self.question_identity_hash != self.operator.question_identity_hash(
            on_key=self.on_key
        ):
            raise ValueError(
                "record question identity must authenticate its operator, "
                "input key, and settings"
            )
        has_absence = (
            self.absence_mode is not None or self.absence_cause is not None
        )
        has_failure = (
            self.failure_type is not None or self.failure_message is not None
        )
        if self.status is RecordStatus.MEASURED:
            if not self.facts:
                raise ValueError("measured records require at least one fact")
            if has_absence or has_failure:
                raise ValueError(
                    "measured records cannot carry absence/failure fields"
                )
            names = [fact.name for fact in self.facts]
            if len(names) != len(set(names)):
                raise ValueError("fact names must be unique within a record")
            if any(
                fact.lineage.evaluation_procedure_config_hash
                != self.evaluation_procedure_config_hash
                for fact in self.facts
            ):
                raise ValueError(
                    "fact lineage must match the record procedure config"
                )
            if any(
                fact.lineage.question_identity_hash
                != self.question_identity_hash
                for fact in self.facts
            ):
                raise ValueError(
                    "fact lineage must match the record question identity"
                )
            if any(
                fact.lineage.operator != self.question for fact in self.facts
            ):
                raise ValueError(
                    "fact lineage operator must match the record question"
                )
            if any(
                fact.lineage.operator_version != self.operator.version
                for fact in self.facts
            ):
                raise ValueError(
                    "fact lineage version must match record operator coordinates"
                )
            return self
        if self.facts:
            raise ValueError("non-measured records cannot carry facts")
        if self.status is RecordStatus.NOT_APPLICABLE:
            if self.absence_mode is None or not self.absence_cause:
                raise ValueError(
                    "not-applicable records require an absence mode and cause"
                )
            if has_failure:
                raise ValueError(
                    "not-applicable records cannot carry failure fields"
                )
            return self
        if not self.failure_type or self.failure_message is None:
            raise ValueError(
                "operator-failure records require failure type and message"
            )
        if has_absence:
            raise ValueError(
                "operator-failure records cannot carry absence fields"
            )
        return self

    def to_current(self) -> NoReturn:
        """Fail closed instead of inventing authenticated identity evidence."""

        raise ValueError(
            "MetricRecord schema 1 cannot be promoted: it lacks operator "
            "implementation, trace producer implementation/source, and sample "
            "identity evidence"
        )


class _MetricRecordV2(FrozenModel):
    """Migration-only PR66 MetricRecord shape."""

    schema_version: Literal[2]
    question: str
    question_identity_hash: str
    on_key: str
    evaluation_procedure_config_hash: EvaluationProcedureConfigHash
    trace_producer: TraceProducer
    sample_identity: SampleIdentity | None = None
    operator: OperatorCoordinates
    status: RecordStatus
    facts: tuple[MetricFact, ...] = ()
    absence_mode: AbsenceMode | None = None
    absence_cause: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None


LoadedMetricRecord: TypeAlias = MetricRecord | ArchivedMetricRecordV1


def load_metric_record(
    payload: PersistedMetricRecordPayload,
) -> LoadedMetricRecord:
    """Read one persisted record, migrating only authenticated schema 2."""

    data = _payload_mapping(payload)
    schema_version = data.get("schema_version")
    if schema_version == 3:
        return MetricRecord.model_validate(data)
    if schema_version == 2:
        previous = _MetricRecordV2.model_validate(data)
        migrated = previous.model_dump(mode="python")
        migrated["schema_version"] = 3
        if previous.status is RecordStatus.NOT_APPLICABLE:
            migrated["failure_code"] = LEGACY_UNSPECIFIED_FAILURE_CODE
        return MetricRecord.model_validate(migrated)
    if schema_version == 1:
        return ArchivedMetricRecordV1.model_validate(data)
    raise ValueError(
        f"unsupported MetricRecord schema version: {schema_version!r}"
    )


def _payload_mapping(
    payload: PersistedMetricRecordPayload,
) -> dict[str, object]:
    if isinstance(payload, (str, bytes, bytearray)):
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("MetricRecord payload must be a JSON object")
        return decoded
    return dict(payload)


__all__ = [
    "ArchivedMetricFactV1",
    "ArchivedMetricRecordV1",
    "ArchivedOperatorCoordinatesV1",
    "ArchivedOperatorLineageV1",
    "LEGACY_UNSPECIFIED_FAILURE_CODE",
    "LoadedMetricRecord",
    "load_metric_record",
]
