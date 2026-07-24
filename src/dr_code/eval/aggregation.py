"""Pure, explicit aggregation over complete input slots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from dr_code.eval.lifecycle import AggregationConfig
from dr_code.models import FrozenModel

StrictFiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class AggregationStatus(StrEnum):
    OK = "ok"
    MISSING_DATA = "missing_data"
    NOT_APPLICABLE = "not_applicable"
    ZERO_DENOMINATOR = "zero_denominator"
    NON_FINITE = "non_finite"


@dataclass(frozen=True, slots=True)
class AggregationInput:
    """Internal value supplied explicitly to an aggregation."""

    value: float | None
    applicable: bool = True


class AggregationOutput(FrozenModel):
    """Serializable, provenance-free aggregation result."""

    status: AggregationStatus
    value: StrictFiniteFloat | None
    count_total: int
    count_applicable: int
    count_present: int


def aggregate(
    config: AggregationConfig,
    inputs: tuple[AggregationInput, ...],
) -> AggregationOutput:
    config = AggregationConfig.model_validate(config.model_dump(mode="python"))
    assignment = config.assignment_dict()
    reduction = assignment["reduction"]
    missing_policy = assignment["missing_data"]
    zero_denominator = assignment["zero_denominator"]
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"unsupported aggregation reduction: {reduction!r}")
    if missing_policy not in {"propagate", "skip"}:
        raise ValueError(
            f"unsupported aggregation missing-data policy: {missing_policy!r}"
        )
    if zero_denominator not in {"not_applicable", "error"}:
        raise ValueError(
            "unsupported aggregation zero-denominator policy: "
            f"{zero_denominator!r}"
        )

    applicable = [item for item in inputs if item.applicable]
    present = [item.value for item in applicable if item.value is not None]
    counts = {
        "count_total": len(inputs),
        "count_applicable": len(applicable),
        "count_present": len(present),
    }
    if not applicable:
        return AggregationOutput(
            status=AggregationStatus.NOT_APPLICABLE,
            value=None,
            **counts,
        )
    if len(present) < len(applicable) and missing_policy == "propagate":
        return AggregationOutput(
            status=AggregationStatus.MISSING_DATA,
            value=None,
            **counts,
        )
    if reduction == "sum":
        try:
            value = float(sum(present))
        except OverflowError:
            value = math.inf
        if not math.isfinite(value):
            return AggregationOutput(
                status=AggregationStatus.NON_FINITE,
                value=None,
                **counts,
            )
        return AggregationOutput(
            status=AggregationStatus.OK,
            value=value,
            **counts,
        )
    if not present:
        if zero_denominator == "error":
            raise ZeroDivisionError(
                "mean aggregation has zero contributing values"
            )
        return AggregationOutput(
            status=AggregationStatus.ZERO_DENOMINATOR,
            value=None,
            **counts,
        )
    try:
        value = sum(present) / len(present)
    except OverflowError:
        value = math.inf
    if not math.isfinite(value):
        return AggregationOutput(
            status=AggregationStatus.NON_FINITE,
            value=None,
            **counts,
        )
    return AggregationOutput(
        status=AggregationStatus.OK,
        value=value,
        **counts,
    )


__all__ = [
    "AggregationInput",
    "AggregationOutput",
    "AggregationStatus",
    "aggregate",
]
