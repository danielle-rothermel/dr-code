"""Absent sentinel with causal lineage."""

from __future__ import annotations

from typing import Literal, TypeIs

from dr_code.models import FrozenModel


class Absent(FrozenModel):
    """A step failed for this input; downstream values inherit the cause.

    Present-but-absent values are data. Preprocessing propagates their lineage,
    and metrics consumers emit not-applicable records instead of raising.
    """

    kind: Literal["absent"] = "absent"
    # instance name that originated the failure
    failed_step: str
    # human-readable reason, stable for lineage joins
    cause: str
    # downstream instance names that inherited it
    propagated_through: tuple[str, ...] = ()


def is_absent(value: object) -> TypeIs[Absent]:
    return isinstance(value, Absent)
