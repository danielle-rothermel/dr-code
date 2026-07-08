"""Inject dead code (unused variables / unreachable statements)."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


class AddDeadCode(InverseTransform):
    """Add an unused variable at module top.

    Recovery is expected via ruff's safe / unsafe fix passes (F841 / F401).
    The injection is intentionally small so the canonical AST still differs
    from ground truth — the contract requires a later normalizer level to
    erase it.
    """

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_DEAD_CODE

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        injected = "import os as _unused_module  # noqa: F401\n"
        return CorruptedSample(
            corrupted_source=injected + source,
        )
