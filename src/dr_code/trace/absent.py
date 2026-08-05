"""Absent sentinel with causal lineage."""

from __future__ import annotations

from typing import Literal, TypeIs

from dr_code.base import FrozenModel


class Absent(FrozenModel):
    """A step failed for this input; downstream values inherit the cause.

    Present-but-absent values are data. Preprocessing propagates their lineage,
    and metrics consumers emit not-applicable records instead of raising.
    """

    kind: Literal["absent"] = "absent"
    # instance name that originated the failure
    failed_step: str
    # machine-readable failure kind, stable for lineage joins and grouping.
    # A plain string here: the vocabulary belongs to the producer that
    # raised, so the trace layer records it without interpreting it.
    failure_code: str
    # human-readable reason, free-form detail for the recorded code
    cause: str
    # downstream instance names that inherited it
    propagated_through: tuple[str, ...] = ()


def is_absent(value: object) -> TypeIs[Absent]:
    return isinstance(value, Absent)
