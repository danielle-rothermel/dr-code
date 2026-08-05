"""Self-describing metric records and dataframe-row flattening.

A record is one comparable answer to one declared metric question. The three
possible answers are three separate models discriminated by ``status``, so a
loaded record carries exactly the fields its answer has and nothing else.

Records validate structure only. They nest registry-free coordinates for the
producer, the metrics definition, and the question they answer, so archived
records stay loadable across settings churn and across operator
implementation and version churn — no registry lookup happens at load. The
guarantee stops at metric-name churn: ``MetricQuestionCoordinate.metric`` is
the closed ``MetricName`` enum, so a record naming a metric that has since
been deleted does not load. The one semantic guarantee they enforce is
internal: a record's question coordinate must appear in the
metrics-definition coordinate it nests.
"""

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
from dr_code.metrics.units import MetricFactUnit
from dr_code.trace import Absent, TraceProducer

METRIC_RECORD_SCHEMA_VERSION: Final = 1


class RecordStatus(StrEnum):
    """The three possible answer shapes for a metric question."""

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    OPERATOR_FAILURE = "operator_failure"


MetricScalar: TypeAlias = float | int | str | bool | None


class MetricFact(FrozenModel):
    """One named, explicitly united observation carried by a record.

    A fact name is non-empty and carries no dot: ``record_rows`` addresses a
    fact's value and unit as ``"{metric}.{name}"`` and
    ``"{metric}.{name}.unit"``, so a fact named ``x.unit`` would occupy fact
    ``x``'s unit column. Banning the separator from names is what makes that
    two-column scheme collision-free.
    """

    name: str
    value: MetricScalar
    unit: MetricFactUnit

    @model_validator(mode="after")
    def reject_non_finite_value(self) -> Self:
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(
                f"metric fact {self.name!r} must be a finite value"
            )
        return self

    @model_validator(mode="after")
    def validate_name(self) -> Self:
        if not self.name:
            raise ValueError("metric fact name must not be empty")
        if "." in self.name:
            raise ValueError(
                f"metric fact name {self.name!r} must not contain '.': "
                "the dot separates a fact column from its unit column"
            )
        return self


class OperatorFailure(FrozenModel):
    """The exception an operator raised while answering its question."""

    failure_type: str
    failure_message: str


class MetricRecordIdentity(FrozenModel):
    """What a record answers, measured by whom, under which declaration."""

    question: MetricQuestionCoordinate
    metric_version: str
    producer: TraceProducer
    metrics_definition: MetricsDefinitionCoordinate

    @model_validator(mode="after")
    def validate_answers_a_declared_question(self) -> Self:
        """The identity must name a question its own definition declares.

        Scoped to internal consistency: the nested definition coordinate is
        the record's own lineage, so its questions must contain this one
        exactly. Deliberately says nothing about the live registry — records
        serialized under older operator versions stay loadable.
        """

        if self.question not in self.metrics_definition.questions:
            raise ValueError(
                f"record for metric {self.question.metric} on "
                f"{self.question.on_key!r} does not match any question in "
                "its metrics_definition"
            )
        return self


class MeasuredRecord(FrozenModel):
    """The operator answered, and the answer is these ordered facts."""

    schema_version: Literal[1] = METRIC_RECORD_SCHEMA_VERSION
    status: Literal[RecordStatus.MEASURED] = RecordStatus.MEASURED
    identity: MetricRecordIdentity
    facts: tuple[MetricFact, ...]

    @model_validator(mode="after")
    def validate_facts(self) -> Self:
        if not self.facts:
            raise ValueError("measured records require at least one fact")
        names = [fact.name for fact in self.facts]
        if len(set(names)) != len(names):
            raise ValueError("measured record fact names must be unique")
        return self


class NotApplicableRecord(FrozenModel):
    """The question had no input to answer, and this is why."""

    schema_version: Literal[1] = METRIC_RECORD_SCHEMA_VERSION
    status: Literal[RecordStatus.NOT_APPLICABLE] = RecordStatus.NOT_APPLICABLE
    identity: MetricRecordIdentity
    absence: Absent


class OperatorFailureRecord(FrozenModel):
    """The operator raised on present input; the failure is the answer."""

    schema_version: Literal[1] = METRIC_RECORD_SCHEMA_VERSION
    status: Literal[RecordStatus.OPERATOR_FAILURE] = (
        RecordStatus.OPERATOR_FAILURE
    )
    identity: MetricRecordIdentity
    failure: OperatorFailure


MetricRecord: TypeAlias = Annotated[
    MeasuredRecord | NotApplicableRecord | OperatorFailureRecord,
    Field(discriminator="status"),
]

#: The one loader for persisted records. ``MetricRecord`` is a discriminated
#: union rather than a class, so deserialization goes through this adapter.
METRIC_RECORD_ADAPTER: Final = TypeAdapter[MetricRecord](MetricRecord)


def record_rows(
    records: Sequence[MetricRecord],
) -> list[dict[str, object]]:
    """Flatten records into sparse rows with collision-free fact columns.

    The identity's own fields are lifted to top-level columns — ``metric``,
    ``on_key``, ``metric_version``, and the composite ``producer``,
    ``metrics_definition``, and ``question_settings`` values — so a row can
    be grouped and joined without reaching into a nested identity mapping.

    Each fact contributes two sibling columns, ``"{metric}.{name}"`` for the
    value and ``"{metric}.{name}.unit"`` for its unit. Two columns per fact
    keeps a row self-describing without multiplying the column space by the
    unit vocabulary, which folding the unit into the value column name would
    do: the same fact measured in two units would then never line up across
    rows. The scheme is collision-free because ``MetricFact`` rejects a dot
    in a fact name: without that rule a fact named ``x.unit`` and fact
    ``x``'s unit column would be the same column.
    """

    rows: list[dict[str, object]] = []
    for record in records:
        row = record.model_dump(mode="python")
        identity = row.pop("identity")
        question = identity.pop("question")
        row["metric"] = question["metric"]
        row["on_key"] = question["on_key"]
        row["question_settings"] = question["settings"]
        row.update(identity)

        facts = row.pop("facts", ())
        prefix = str(record.identity.question.metric)
        for fact in facts:
            column = f"{prefix}.{fact['name']}"
            row[column] = fact["value"]
            row[f"{column}.unit"] = fact["unit"]
        rows.append(row)
    return rows


__all__ = [
    "METRIC_RECORD_ADAPTER",
    "METRIC_RECORD_SCHEMA_VERSION",
    "MeasuredRecord",
    "MetricFact",
    "MetricRecord",
    "MetricRecordIdentity",
    "MetricScalar",
    "NotApplicableRecord",
    "OperatorFailure",
    "OperatorFailureRecord",
    "RecordStatus",
]
