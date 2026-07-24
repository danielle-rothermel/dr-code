"""Canonical Metric Facts, Records, Scores, and operator lineage."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Final, Literal, Self, TypeAlias, cast

from pydantic import (
    AfterValidator,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_code.eval.identity import (
    SCHEMA_METRIC_QUESTION_BINDING,
    identity_hash_for,
)
from dr_code.trace.observation import SampleIdentity
from dr_code.eval.immutable_json import freeze_json, thaw_json
from dr_code.models import FrozenModel
from dr_code.trace import TraceProducer

StrictFiniteFloat = Annotated[
    float,
    Field(strict=True, allow_inf_nan=False),
]


def validate_sha256(value: str, *, owner: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{owner} must be a lowercase SHA-256")
    return value


def validate_evaluation_procedure_config_hash(value: str) -> str:
    try:
        validate_sha256(value, owner="evaluation procedure config hash")
    except ValueError:
        raise ValueError(
            "evaluation procedure config hash must be a lowercase SHA-256"
        ) from None
    return value


EvaluationProcedureConfigHash = Annotated[
    StrictStr,
    AfterValidator(validate_evaluation_procedure_config_hash),
]
FactScalar: TypeAlias = StrictFiniteFloat | StrictInt | StrictStr | StrictBool
FactValue: TypeAlias = FactScalar | None
METRIC_RECORD_SCHEMA_VERSION: Final = 3
SCORE_SCHEMA_VERSION: Final = 1


def _validate_fact_scalar(value: object) -> object:
    if type(value) not in (float, int, str, bool):
        raise ValueError("fact values must be strict JSON scalars")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("fact values must be finite")
    return value


class Applicability(StrEnum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class AbsenceMode(StrEnum):
    PREPROCESSING_FAILURE = "preprocessing_failure"
    NO_INPUT = "no_input"
    NO_TRACE = "no_trace"
    MISSING_TRACE_KEY = "missing_trace_key"
    EMPTY_CANDIDATE_SET = "empty_candidate_set"


class OperatorLineage(FrozenModel):
    """Resolved implementation coordinates for a fact."""

    evaluation_procedure_config_hash: EvaluationProcedureConfigHash
    question_identity_hash: str
    operator: str
    operator_version: str
    operator_implementation: str
    step: str | None = None
    step_version: str | None = None

    @model_validator(mode="after")
    def validate_step_pair(self) -> Self:
        validate_sha256(
            self.operator_implementation,
            owner="operator implementation",
        )
        if (self.step is None) != (self.step_version is None):
            raise ValueError("step and step_version must be set together")
        return self


class OperatorCoordinates(FrozenModel):
    """Resolved operator implementation and fully validated settings."""

    name: str
    version: str
    implementation_hash: str
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
        validate_sha256(
            self.implementation_hash,
            owner="operator implementation",
        )
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


class RecordStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    OPERATOR_FAILURE = "operator_failure"


class MetricRecord(FrozenModel):
    """Exactly one answer shape for one metric question."""

    schema_version: Literal[3]
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
    failure_code: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    def fact_values(self) -> dict[str, FactValue]:
        """Return the record's facts as a name-to-value projection."""

        return {fact.name: fact.value for fact in self.facts}

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
            self.absence_mode is not None
            or self.absence_cause is not None
            or self.failure_code is not None
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
            if any(
                fact.lineage.operator_implementation
                != self.operator.implementation_hash
                for fact in self.facts
            ):
                raise ValueError(
                    "fact lineage implementation must match record operator "
                    "coordinates"
                )
            return self
        if self.facts:
            raise ValueError("non-measured records cannot carry facts")
        if self.status is RecordStatus.NOT_APPLICABLE:
            if (
                self.absence_mode is None
                or not self.absence_cause
                or not self.failure_code
            ):
                raise ValueError(
                    "not-applicable records require an absence mode, cause, and failure code"
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

    @classmethod
    def measured(
        cls,
        *,
        question: str,
        question_identity_hash: str,
        on_key: str,
        evaluation_procedure_config_hash: EvaluationProcedureConfigHash,
        trace_producer: TraceProducer,
        operator: OperatorCoordinates,
        facts: tuple[MetricFact, ...],
        sample_identity: SampleIdentity | None = None,
    ) -> Self:
        return cls(
            schema_version=METRIC_RECORD_SCHEMA_VERSION,
            question=question,
            question_identity_hash=question_identity_hash,
            on_key=on_key,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            trace_producer=trace_producer,
            sample_identity=sample_identity,
            operator=operator,
            status=RecordStatus.MEASURED,
            facts=facts,
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        question: str,
        question_identity_hash: str,
        on_key: str,
        evaluation_procedure_config_hash: EvaluationProcedureConfigHash,
        trace_producer: TraceProducer,
        operator: OperatorCoordinates,
        absence_mode: AbsenceMode,
        cause: str,
        failure_code: str,
        sample_identity: SampleIdentity | None = None,
    ) -> Self:
        return cls(
            schema_version=METRIC_RECORD_SCHEMA_VERSION,
            question=question,
            question_identity_hash=question_identity_hash,
            on_key=on_key,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            trace_producer=trace_producer,
            sample_identity=sample_identity,
            operator=operator,
            status=RecordStatus.NOT_APPLICABLE,
            absence_mode=absence_mode,
            absence_cause=cause,
            failure_code=failure_code,
        )

    @classmethod
    def operator_failure(
        cls,
        *,
        question: str,
        question_identity_hash: str,
        on_key: str,
        evaluation_procedure_config_hash: EvaluationProcedureConfigHash,
        trace_producer: TraceProducer,
        operator: OperatorCoordinates,
        failure_type: str,
        failure_message: str,
        sample_identity: SampleIdentity | None = None,
    ) -> Self:
        return cls(
            schema_version=METRIC_RECORD_SCHEMA_VERSION,
            question=question,
            question_identity_hash=question_identity_hash,
            on_key=on_key,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            trace_producer=trace_producer,
            sample_identity=sample_identity,
            operator=operator,
            status=RecordStatus.OPERATOR_FAILURE,
            failure_type=failure_type,
            failure_message=failure_message,
        )


class Score(FrozenModel):
    """A deterministic derivation from named Metric Facts."""

    schema_version: Literal[1]
    name: str
    value: FactScalar
    unit: str
    evaluation_procedure_config_hash: EvaluationProcedureConfigHash
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


def record_rows(
    records: Sequence[MetricRecord],
) -> list[dict[str, object]]:
    """Flatten records into sparse rows with question-prefixed facts."""

    rows: list[dict[str, object]] = []
    for record in records:
        row = record.model_dump(mode="python")
        facts = row.pop("facts")
        assert isinstance(facts, tuple)
        for fact in record.facts:
            row[f"{record.question}.{fact.name}"] = fact.value
        rows.append(row)
    return rows


__all__ = [
    "AbsenceMode",
    "Applicability",
    "FactScalar",
    "FactValue",
    "METRIC_RECORD_SCHEMA_VERSION",
    "MetricFact",
    "MetricRecord",
    "OperatorLineage",
    "OperatorCoordinates",
    "RecordStatus",
    "SCORE_SCHEMA_VERSION",
    "Score",
    "record_rows",
]
