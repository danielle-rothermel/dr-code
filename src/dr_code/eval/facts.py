"""Metric Facts, Metric Records, and Scores with explicit lineage.

- A :class:`MetricFact` is a measured named value with an **explicit
  unit**, an **applicability** state, and lineage: the Evaluation
  Procedure Config identity plus resolved operator/step lineage.
- A :class:`MetricRecord` is the self-describing answer to one Metric
  Question: it carries facts, or an explicit not-applicable state, or an
  explicit operator-failure state.
- A :class:`Score` is deterministically derived from Facts under one
  Evaluation Procedure Config, retaining derivation lineage.

The distinguishable *absence* modes (proved separately in tests) are:
native ``Absent`` (a causal Preprocessing Failure on a *present* input),
no-input, no-trace, missing-trace-key, and empty-candidate-set. Only the
first is a Preprocessing Failure; the rest are their own explicit values.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self, TypeAlias

from pydantic import model_validator

from dr_code.models import FrozenModel

FactScalar: TypeAlias = float | int | str | bool


class Applicability(StrEnum):
    """Whether a fact's measurement applies to its input."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class AbsenceMode(StrEnum):
    """The distinguishable reasons a measurement is absent.

    ``PREPROCESSING_FAILURE`` is the native ``Absent`` role: a causal
    failure while processing a *present* input. The rest are explicitly
    distinct and MUST NOT be collapsed onto it.
    """

    PREPROCESSING_FAILURE = "preprocessing_failure"
    NO_INPUT = "no_input"
    NO_TRACE = "no_trace"
    MISSING_TRACE_KEY = "missing_trace_key"
    EMPTY_CANDIDATE_SET = "empty_candidate_set"


class OperatorLineage(FrozenModel):
    """Resolved operator/step lineage attached to a fact."""

    evaluation_procedure_config_hash: str
    operator: str
    operator_version: str
    step: str | None = None
    step_version: str | None = None


class MetricFact(FrozenModel):
    """A measured named value with explicit unit, applicability, lineage."""

    name: str
    value: FactScalar
    unit: str
    applicability: Applicability
    lineage: OperatorLineage

    @model_validator(mode="after")
    def _unit_present(self) -> Self:
        if self.unit == "":
            raise ValueError("a metric fact requires an explicit unit")
        return self


class RecordStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    OPERATOR_FAILURE = "operator_failure"


class MetricRecord(FrozenModel):
    """Self-describing answer to one Metric Question.

    Exactly one of the three shapes is populated:
    - ``MEASURED``: one or more facts, no absence/failure fields.
    - ``NOT_APPLICABLE``: an explicit :class:`AbsenceMode`, no facts.
    - ``OPERATOR_FAILURE``: a failure type + message, no facts.
    """

    question: str
    on_key: str
    evaluation_procedure_config_hash: str
    status: RecordStatus
    facts: tuple[MetricFact, ...] = ()
    absence_mode: AbsenceMode | None = None
    absence_cause: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> Self:
        if self.status is RecordStatus.MEASURED:
            if not self.facts:
                raise ValueError("measured records require at least one fact")
            if self.absence_mode is not None or self.failure_type is not None:
                raise ValueError(
                    "measured records cannot carry absence/failure fields"
                )
            return self

        if self.facts:
            raise ValueError("non-measured records cannot carry facts")

        if self.status is RecordStatus.NOT_APPLICABLE:
            if self.absence_mode is None or self.absence_cause is None:
                raise ValueError(
                    "not-applicable records require an absence mode and cause"
                )
            if self.failure_type is not None:
                raise ValueError(
                    "not-applicable records cannot carry failure fields"
                )
            return self

        # OPERATOR_FAILURE
        if not self.failure_type or not self.failure_message:
            raise ValueError(
                "operator-failure records require a failure type and message"
            )
        if self.absence_mode is not None:
            raise ValueError(
                "operator-failure records cannot carry absence fields"
            )
        return self

    @classmethod
    def measured(
        cls,
        *,
        question: str,
        on_key: str,
        evaluation_procedure_config_hash: str,
        facts: tuple[MetricFact, ...],
    ) -> Self:
        return cls(
            question=question,
            on_key=on_key,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            status=RecordStatus.MEASURED,
            facts=facts,
        )

    @classmethod
    def not_applicable(
        cls,
        *,
        question: str,
        on_key: str,
        evaluation_procedure_config_hash: str,
        absence_mode: AbsenceMode,
        cause: str,
    ) -> Self:
        return cls(
            question=question,
            on_key=on_key,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            status=RecordStatus.NOT_APPLICABLE,
            absence_mode=absence_mode,
            absence_cause=cause,
        )

    @classmethod
    def operator_failure(
        cls,
        *,
        question: str,
        on_key: str,
        evaluation_procedure_config_hash: str,
        failure_type: str,
        failure_message: str,
    ) -> Self:
        return cls(
            question=question,
            on_key=on_key,
            evaluation_procedure_config_hash=evaluation_procedure_config_hash,
            status=RecordStatus.OPERATOR_FAILURE,
            failure_type=failure_type,
            failure_message=failure_message,
        )


class Score(FrozenModel):
    """A named value deterministically derived from Metric Facts.

    Retains derivation lineage: the Evaluation Procedure Config identity
    and the source fact names it was derived from.
    """

    name: str
    value: FactScalar
    unit: str
    evaluation_procedure_config_hash: str
    derived_from: tuple[str, ...]

    @model_validator(mode="after")
    def _unit_present(self) -> Self:
        if self.unit == "":
            raise ValueError("a score requires an explicit unit")
        return self


__all__ = [
    "AbsenceMode",
    "Applicability",
    "FactScalar",
    "MetricFact",
    "MetricRecord",
    "OperatorLineage",
    "RecordStatus",
    "Score",
]
