"""Wrap source in a single-backtick inline span."""

from __future__ import annotations

import random
from typing import ClassVar

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import ExtractorName, InverseTransformName
from code_eval.synthetic.inverse_transforms.base import InverseTransform


class AddInlineBackticks(InverseTransform):
    """Wrap the entire source in a single-backtick inline span."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_INLINE_BACKTICKS
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {ExtractorName.INLINE_SPANS.value}
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=f"`{source.rstrip()}`",
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
        )
