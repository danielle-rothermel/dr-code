from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, verify, UNIQUE
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import EvaluationCandidateIdentity
from dr_code.evaluation.plan import (
    AggregationPolicy,
    AggregationStatistic,
    NotApplicablePolicy,
)
from dr_code.metrics import (
    MeasuredRecord,
    MetricRecord,
    NotApplicableRecord,
    OperatorFailureRecord,
)


class AggregationSlot(FrozenModel):
    candidate: EvaluationCandidateIdentity
    record: MetricRecord | None = None


class AggregationInput(FrozenModel):
    policy: AggregationPolicy
    slots: tuple[AggregationSlot, ...]

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        if not self.slots:
            raise ValueError("an aggregation input requires at least one slot")
        candidates = [slot.candidate for slot in self.slots]
        if len(set(candidates)) != len(candidates):
            raise ValueError(
                "aggregation slots must name distinct candidate coordinates"
            )
        return self


@verify(UNIQUE)
class AggregationStatus(StrEnum):
    # Never build payloads by iterating this closed vocabulary.

    OK = "ok"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    EMPTY_DENOMINATOR = "empty_denominator"
    NON_FINITE = "non_finite"


class AggregationOk(FrozenModel):
    status: Literal[AggregationStatus.OK] = AggregationStatus.OK
    value: float
    counted: int = Field(ge=0)
    excluded: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError("an ok aggregation value must be finite")
        return self


class AggregationMissing(FrozenModel):
    status: Literal[AggregationStatus.MISSING] = AggregationStatus.MISSING
    missing: tuple[EvaluationCandidateIdentity, ...]

    @model_validator(mode="after")
    def validate_missing(self) -> Self:
        if not self.missing:
            raise ValueError("a missing result must name the empty slots")
        return self


class AggregationNotApplicable(FrozenModel):
    status: Literal[AggregationStatus.NOT_APPLICABLE] = (
        AggregationStatus.NOT_APPLICABLE
    )
    refused: tuple[EvaluationCandidateIdentity, ...]

    @model_validator(mode="after")
    def validate_refused(self) -> Self:
        if not self.refused:
            raise ValueError(
                "a not-applicable result must name the refusing slots"
            )
        return self


class AggregationEmptyDenominator(FrozenModel):
    status: Literal[AggregationStatus.EMPTY_DENOMINATOR] = (
        AggregationStatus.EMPTY_DENOMINATOR
    )
    excluded: int

    @model_validator(mode="after")
    def validate_excluded(self) -> Self:
        if self.excluded < 1:
            raise ValueError(
                "an empty-denominator result excludes at least one slot"
            )
        return self


class AggregationNonFinite(FrozenModel):
    status: Literal[AggregationStatus.NON_FINITE] = (
        AggregationStatus.NON_FINITE
    )
    counted: int = Field(ge=0)
    reason: str = Field(min_length=1)


AggregationResult: TypeAlias = Annotated[
    AggregationOk
    | AggregationMissing
    | AggregationNotApplicable
    | AggregationEmptyDenominator
    | AggregationNonFinite,
    Field(discriminator="status"),
]


@dataclass(frozen=True, slots=True)
class _Contributions:
    values: tuple[float, ...]
    excluded: int


def aggregate(request: AggregationInput) -> AggregationResult:
    policy = request.policy
    missing = tuple(
        slot.candidate for slot in request.slots if slot.record is None
    )
    if missing:
        return AggregationMissing(missing=missing)

    refused: list[EvaluationCandidateIdentity] = []
    values: list[float] = []
    excluded = 0
    for slot in request.slots:
        record = slot.record
        assert record is not None
        outcome = _slot_contribution(record, policy)
        match outcome:
            case _Refused():
                refused.append(slot.candidate)
            case _Excluded():
                excluded += 1
            case _Counted(value=value):
                values.append(value)
    if refused:
        return AggregationNotApplicable(refused=tuple(refused))

    return _reduce(
        _Contributions(values=tuple(values), excluded=excluded), policy
    )


@dataclass(frozen=True, slots=True)
class _Counted:
    value: float


@dataclass(frozen=True, slots=True)
class _Excluded:
    pass


@dataclass(frozen=True, slots=True)
class _Refused:
    pass


_SlotOutcome: TypeAlias = _Counted | _Excluded | _Refused


def _slot_contribution(
    record: MetricRecord, policy: AggregationPolicy
) -> _SlotOutcome:
    if record.identity.question != policy.question:
        raise ValueError(
            f"record answers metric {record.identity.question.metric} on "
            f"{record.identity.question.on_key!r}, but the policy "
            f"aggregates {policy.question.metric} on "
            f"{policy.question.on_key!r}"
        )
    if isinstance(record, NotApplicableRecord):
        return _by_policy(policy.not_applicable)
    if isinstance(record, OperatorFailureRecord):
        return _by_policy(policy.operator_failure)
    return _measured_contribution(record, policy)


def _by_policy(rule: NotApplicablePolicy) -> _SlotOutcome:
    match rule:
        case NotApplicablePolicy.EXCLUDE:
            return _Excluded()
        case NotApplicablePolicy.ZERO:
            return _Counted(value=0.0)
        case NotApplicablePolicy.FAIL:
            return _Refused()


def _measured_contribution(
    record: MeasuredRecord, policy: AggregationPolicy
) -> _SlotOutcome:
    for value in record.values:
        if value.name == policy.value:
            return _Counted(value=_numeric(value.name, value.value))
    raise ValueError(
        f"measured record carries no value named {policy.value!r}"
    )


def _numeric(name: str, value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        try:
            return float(value)
        except OverflowError:
            return math.inf if value > 0 else -math.inf
    raise ValueError(
        f"metric value {name!r} has non-numeric value {value!r} and cannot be "
        "aggregated"
    )


def _non_finite(
    policy: AggregationPolicy, counted: int, problem: str
) -> AggregationNonFinite:
    return AggregationNonFinite(
        counted=counted,
        reason=f"{policy.statistic.value} of {counted} values {problem}",
    )


def _reduce(
    contributions: _Contributions, policy: AggregationPolicy
) -> AggregationResult:
    values = contributions.values
    if policy.statistic is AggregationStatistic.COUNT:
        return AggregationOk(
            value=float(len(values)),
            counted=len(values),
            excluded=contributions.excluded,
        )
    if not values:
        return AggregationEmptyDenominator(excluded=contributions.excluded)

    # fsum overflow and opposite infinities are non-finite results.
    try:
        match policy.statistic:
            case AggregationStatistic.SUM:
                value = math.fsum(values)
            case AggregationStatistic.MEAN:
                value = math.fsum(values) / len(values)
            case AggregationStatistic.PROPORTION:
                truthy = sum(1 for item in values if item != 0.0)
                value = truthy / len(values)
            case AggregationStatistic.COUNT:  # pragma: no cover
                raise AssertionError("count is reduced before this point")
    except OverflowError:
        return _non_finite(policy, len(values), "overflows a float")
    except ValueError:
        return _non_finite(policy, len(values), "mixes opposite infinities")

    if not math.isfinite(value):
        return _non_finite(policy, len(values), "is not finite")
    return AggregationOk(
        value=value,
        counted=len(values),
        excluded=contributions.excluded,
    )


__all__ = [
    "AggregationEmptyDenominator",
    "AggregationInput",
    "AggregationMissing",
    "AggregationNonFinite",
    "AggregationNotApplicable",
    "AggregationOk",
    "AggregationResult",
    "AggregationSlot",
    "AggregationStatus",
    "aggregate",
]
