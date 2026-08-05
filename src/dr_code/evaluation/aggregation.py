"""Pure reduction of metric records to one aggregate value.

``aggregate`` is entirely pure: it reads its inputs and returns a result.
No I/O, no registry lookups, no clock, no randomness — the same input
always produces the same output, so an aggregate can be recomputed from
archived records years later and must come out identical.

Inputs are *complete explicit slots*. Every slot the plan expected appears
in the input, and a slot with no record says so by carrying ``None``, rather
than being silently absent from a shorter list. That makes the four ways an
aggregate can fail to be a number genuinely distinct and separately typed:

- ``missing`` — a planned slot produced no record at all; the run is
  incomplete and the aggregate is not defined.
- ``not_applicable`` — a record said the question had no input, under a
  policy that refuses to count it.
- ``empty_denominator`` — every slot was legitimately excluded, so the
  statistic divides by nothing.
- ``non_finite`` — the arithmetic completed but produced no usable number,
  whether it overflowed or evaluated to an infinity or a NaN.

None of these is a sentinel float. A caller pattern-matches on the result
type; there is no in-band value to mistake for a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, verify, UNIQUE
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import Field, model_validator

from dr_code.base import FrozenModel
from dr_code.evaluation.coordinates import CandidateCoordinate
from dr_code.evaluation.plan import (
    AggregationPolicy,
    AggregationStatistic,
    NotApplicablePolicy,
)
from dr_code.metrics import (
    MeasuredRecord,
    MetricQuestionCoordinate,
    MetricRecord,
    NotApplicableRecord,
    OperatorFailureRecord,
)


class AggregationSlot(FrozenModel):
    """One planned measurement position and whatever filled it.

    ``record`` is ``None`` when the slot produced nothing. That is a
    distinct state from a record reporting not-applicable: nothing ran, as
    against something ran and reported that the question did not apply.
    """

    candidate: CandidateCoordinate
    record: MetricRecord | None = None


class AggregationInput(FrozenModel):
    """The complete set of slots one aggregation reduces.

    Completeness is the caller's assertion and this model's invariant: the
    slots are exactly the positions the plan expected, each named once.
    ``aggregate`` reduces what it is given and never infers which slots
    should have existed.
    """

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
    """Whether an aggregation produced a value, and if not why not.

    Never build a payload by iterating this enum.
    """

    OK = "ok"
    MISSING = "missing"
    NOT_APPLICABLE = "not_applicable"
    EMPTY_DENOMINATOR = "empty_denominator"
    NON_FINITE = "non_finite"


class FactCoordinate(FrozenModel):
    """The complete address of one metric fact.

    A question coordinate names the measurement; the fact name names which
    of that measurement's numbers. Together they are the full coordinate.
    An empty fact name addresses nothing, and is rejected here on the same
    terms ``AggregationPolicy`` rejects it: same-shaped address, same
    strictness. A dotted name is rejected for the same reason
    ``MetricFact`` rejects one — no real fact can carry it, so such a
    coordinate could only ever address a fact that cannot exist.
    """

    question: MetricQuestionCoordinate
    fact: str

    @model_validator(mode="after")
    def validate_fact_name(self) -> Self:
        if not self.fact:
            raise ValueError("a fact coordinate must name a fact")
        if "." in self.fact:
            raise ValueError(
                f"fact name {self.fact!r} must not contain '.': no metric "
                "fact can carry a dotted name"
            )
        return self


class AggregationOk(FrozenModel):
    """The reduction produced a finite value from these slots."""

    status: Literal[AggregationStatus.OK] = AggregationStatus.OK
    value: float
    #: How many slots contributed to the numerator and denominator.
    counted: int
    #: Every slot excluded by policy rather than counted.
    excluded: int

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if not math.isfinite(self.value):
            raise ValueError("an ok aggregation value must be finite")
        return self


class AggregationMissing(FrozenModel):
    """Planned slots produced no record, so the aggregate is undefined."""

    status: Literal[AggregationStatus.MISSING] = AggregationStatus.MISSING
    #: The empty slots, in input order.
    missing: tuple[CandidateCoordinate, ...]

    @model_validator(mode="after")
    def validate_missing(self) -> Self:
        if not self.missing:
            raise ValueError("a missing result must name the empty slots")
        return self


class AggregationNotApplicable(FrozenModel):
    """A slot reported no applicable input under a policy that refuses it."""

    status: Literal[AggregationStatus.NOT_APPLICABLE] = (
        AggregationStatus.NOT_APPLICABLE
    )
    #: The refusing slots, in input order.
    refused: tuple[CandidateCoordinate, ...]

    @model_validator(mode="after")
    def validate_refused(self) -> Self:
        if not self.refused:
            raise ValueError(
                "a not-applicable result must name the refusing slots"
            )
        return self


class AggregationEmptyDenominator(FrozenModel):
    """Every slot was excluded, so the statistic has nothing to divide."""

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
    """The arithmetic completed but did not produce a finite number."""

    status: Literal[AggregationStatus.NON_FINITE] = (
        AggregationStatus.NON_FINITE
    )
    counted: int
    #: Why the value is not finite, as a plain description.
    reason: str


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
    """Internal tally of what the slots contributed, before reduction."""

    values: tuple[float, ...]
    excluded: int


def aggregate(request: AggregationInput) -> AggregationResult:
    """Reduce complete slots to one typed aggregation result.

    Discrimination over the record union is a type check, never an
    inspection of nullable fields: a record's class *is* its outcome.
    """

    policy = request.policy
    missing = tuple(
        slot.candidate for slot in request.slots if slot.record is None
    )
    if missing:
        return AggregationMissing(missing=missing)

    refused: list[CandidateCoordinate] = []
    values: list[float] = []
    excluded = 0
    for slot in request.slots:
        record = slot.record
        assert record is not None  # every empty slot returned above
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
    """The slot contributes this value to the statistic."""

    value: float


@dataclass(frozen=True, slots=True)
class _Excluded:
    """The slot leaves the denominator by policy."""


@dataclass(frozen=True, slots=True)
class _Refused:
    """The slot invalidates the aggregate by policy."""


_SlotOutcome: TypeAlias = _Counted | _Excluded | _Refused


def _slot_contribution(
    record: MetricRecord, policy: AggregationPolicy
) -> _SlotOutcome:
    """Decide what one record contributes, by its type and the policy.

    The question check precedes the type dispatch because answering a
    different question is a mismatch between the aggregation and its
    input for every record type, not only for measured ones: a
    not-applicable or operator-failure record about some other metric
    would otherwise silently exclude or zero-fill a slot.
    """

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
    """Apply a denominator rule to a non-measured record."""

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
    """Read the policy's fact out of a measured record.

    The caller has already established that the record answers the
    policy's question. A record that carries no fact by the policy's name
    is a mismatch between the aggregation and its input rather than a
    measurement outcome, so it raises rather than quietly becoming an
    excluded slot.
    """

    for fact in record.facts:
        if fact.name == policy.fact:
            return _Counted(value=_numeric(fact.name, fact.value))
    raise ValueError(f"measured record carries no fact named {policy.fact!r}")


def _numeric(name: str, value: object) -> float:
    """Coerce a metric scalar to the float the statistics operate on.

    ``MetricFact`` accepts an arbitrarily large ``int``, so a persisted
    record can carry a value that no float represents. Coercing it raises
    ``OverflowError``, which would escape ``aggregate`` from outside the
    guarded reduction; it is converted to an infinity here instead, so the
    reduction's finiteness check reports it as a non-finite result exactly
    as it reports an overflowing ``fsum``.
    """

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        try:
            return float(value)
        except OverflowError:
            return math.inf if value > 0 else -math.inf
    raise ValueError(
        f"fact {name!r} has non-numeric value {value!r} and cannot be "
        "aggregated"
    )


def _non_finite(
    policy: AggregationPolicy, counted: int, problem: str
) -> AggregationNonFinite:
    """Describe why a completed reduction is not a usable number."""

    return AggregationNonFinite(
        counted=counted,
        reason=f"{policy.statistic.value} of {counted} values {problem}",
    )


def _reduce(
    contributions: _Contributions, policy: AggregationPolicy
) -> AggregationResult:
    """Apply the policy's statistic to the counted values."""

    values = contributions.values
    if policy.statistic is AggregationStatistic.COUNT:
        return AggregationOk(
            value=float(len(values)),
            counted=len(values),
            excluded=contributions.excluded,
        )
    if not values:
        return AggregationEmptyDenominator(excluded=contributions.excluded)

    # ``math.fsum`` raises rather than returning a number in two cases:
    # ``OverflowError`` when the exact sum overflows a float, and
    # ``ValueError`` ("-inf + inf in fsum") when the values include both
    # infinities — reachable here because ``_numeric`` maps oversized
    # persisted ints of either sign to an infinity. Both are arithmetic
    # that completed without producing a number, which is exactly what
    # the non-finite result is for, so both are caught here rather than
    # escaping as an exception: a caller pattern-matching the result must
    # not also have to guard the call.
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
    "FactCoordinate",
    "aggregate",
]
