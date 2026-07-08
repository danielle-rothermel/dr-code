"""Wrap source in a single-backtick inline span."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


class AddInlineBackticks(InverseTransform):
    """Wrap the entire source in a single-backtick inline span."""

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_INLINE_BACKTICKS
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=f"`{source.rstrip()}`",
        )
