"""Absent sentinel with causal lineage."""

from __future__ import annotations

from typing import Final, Literal, TypeIs

from dr_code.models import FrozenModel

LEGACY_FAILURE_CODE: Final = "legacy.unknown"


class Absent(FrozenModel):
    """A step failed for this input; downstream values inherit the cause.

    Present-but-Absent is data (eval-flow L2): consumers emit
    not-applicable records instead of raising.
    """

    kind: Literal["absent"] = "absent"
    # instance name that originated the failure
    failed_step: str
    # human-readable reason for the failure
    cause: str
    # stable machine-readable failure category. Old serialized traces did
    # not carry this field, so they materialize as this explicit legacy value.
    failure_code: str = LEGACY_FAILURE_CODE
    # downstream instance names that inherited it
    propagated_through: tuple[str, ...] = ()


def is_absent(value: object) -> TypeIs[Absent]:
    return isinstance(value, Absent)
