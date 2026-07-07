"""Inject dead code (unused variables / unreachable statements)."""

from __future__ import annotations

import random
from typing import ClassVar

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import InverseTransformName, NormalizerName
from code_eval.synthetic.inverse_transforms.base import InverseTransform


class AddDeadCode(InverseTransform):
    """Add an unused variable at module top.

    Recovery is expected via ruff's safe / unsafe fix passes (F841 / F401).
    The injection is intentionally small so the canonical AST still differs
    from ground truth — the contract requires a later normalizer level to
    erase it.
    """

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_DEAD_CODE
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {
            NormalizerName.L3_RUFF_FIX_SAFE.value,
            NormalizerName.L4_RUFF_FIX_UNSAFE.value,
        }
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        injected = "import os as _unused_module  # noqa: F401\n"
        return CorruptedSample(
            corrupted_source=injected + source,
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
        )
