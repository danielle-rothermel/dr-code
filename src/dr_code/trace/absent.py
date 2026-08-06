from __future__ import annotations

from typing import Literal, TypeIs

from dr_code.core.models import FrozenModel


class Absent(FrozenModel):
    kind: Literal["absent"] = "absent"
    failed_step: str
    failure_code: str
    cause: str
    propagated_through: tuple[str, ...] = ()


def is_absent(value: object) -> TypeIs[Absent]:
    return isinstance(value, Absent)
