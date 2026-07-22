"""Pure aggregation: an Aggregation Config as a deterministic function.

:func:`aggregate` applies an :class:`AggregationConfig` over an
*explicitly complete* collection of numeric input values and returns a
provenance-free :class:`AggregationOutput`. The output carries explicit
missing-data / applicability / zero-denominator values and NO lifecycle
Result, authority, orchestration, Objective, or Reward semantics.

The function is pure and deterministic: same Config + same ordered inputs
=> same output, with no I/O, clocks, or ambient state.
"""

from __future__ import annotations

from enum import StrEnum

from dr_code.eval.lifecycle import AggregationConfig
from dr_code.models import FrozenModel


class AggregationStatus(StrEnum):
    """The explicit outcome status of an aggregation."""

    OK = "ok"
    MISSING_DATA = "missing_data"
    NOT_APPLICABLE = "not_applicable"
    ZERO_DENOMINATOR = "zero_denominator"


class AggregationInput(FrozenModel):
    """One explicitly complete input slot.

    ``value`` is ``None`` when the slot is missing data; ``applicable``
    is ``False`` when the slot is not applicable. Both are explicit
    rather than inferred from absence.
    """

    value: float | None
    applicable: bool = True


class AggregationOutput(FrozenModel):
    """Provenance-free result of one aggregation.

    Carries the numeric ``value`` (``None`` for any non-OK status), the
    explicit ``status``, and the counts that produced it. No lifecycle,
    authority, orchestration, Objective, or Reward fields.
    """

    status: AggregationStatus
    value: float | None
    count_total: int
    count_applicable: int
    count_present: int


def aggregate(
    config: AggregationConfig,
    inputs: tuple[AggregationInput, ...],
) -> AggregationOutput:
    """Reduce ``inputs`` under ``config`` into a pure output.

    Behavior is driven entirely by the Config assignment:
    - ``missing_data = "propagate"``: any missing present value yields a
      MISSING_DATA output; ``"skip"`` drops missing slots.
    - Not-applicable slots are always excluded from the reduction; a
      wholly not-applicable input yields NOT_APPLICABLE.
    - A ``mean`` over zero contributing slots yields ZERO_DENOMINATOR
      (unless ``zero_denominator = "error"``, which raises).
    """

    assignment = config.assignment_dict()
    reduction = assignment["reduction"]
    missing_policy = assignment.get("missing_data", "propagate")
    zero_denominator = assignment.get("zero_denominator", "not_applicable")

    count_total = len(inputs)
    applicable = [item for item in inputs if item.applicable]
    count_applicable = len(applicable)

    if count_applicable == 0:
        return AggregationOutput(
            status=AggregationStatus.NOT_APPLICABLE,
            value=None,
            count_total=count_total,
            count_applicable=0,
            count_present=0,
        )

    present = [item.value for item in applicable if item.value is not None]
    count_present = len(present)
    has_missing = count_present < count_applicable

    if has_missing and missing_policy == "propagate":
        return AggregationOutput(
            status=AggregationStatus.MISSING_DATA,
            value=None,
            count_total=count_total,
            count_applicable=count_applicable,
            count_present=count_present,
        )

    if reduction == "sum":
        return AggregationOutput(
            status=AggregationStatus.OK,
            value=float(sum(present)),
            count_total=count_total,
            count_applicable=count_applicable,
            count_present=count_present,
        )

    # mean
    if count_present == 0:
        if zero_denominator == "error":
            raise ZeroDivisionError(
                "mean aggregation has zero contributing values"
            )
        return AggregationOutput(
            status=AggregationStatus.ZERO_DENOMINATOR,
            value=None,
            count_total=count_total,
            count_applicable=count_applicable,
            count_present=0,
        )

    return AggregationOutput(
        status=AggregationStatus.OK,
        value=sum(present) / count_present,
        count_total=count_total,
        count_applicable=count_applicable,
        count_present=count_present,
    )


__all__ = [
    "AggregationInput",
    "AggregationOutput",
    "AggregationStatus",
    "aggregate",
]
