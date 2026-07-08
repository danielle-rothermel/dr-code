"""Wrap source twice in code fences to simulate "Option A / Option B"."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform

_FENCE: Final[str] = "```python"
_CLOSE: Final[str] = "```"


class AddMultipleSolutions(InverseTransform):
    """Emit a "Option 1 / Option 2" message with two fenced blocks.

    The first block is the real source. The second block is a trivial
    placeholder. The recovery contract just requires the fence extractor
    to find both candidates — the validator picks the one that parses.
    """

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_MULTIPLE_SOLUTIONS
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        alt = "def _alt_solution():\n    raise NotImplementedError\n"
        wrapped = (
            "Option 1:\n"
            f"{_FENCE}\n{source.rstrip()}\n{_CLOSE}\n\n"
            "Option 2:\n"
            f"{_FENCE}\n{alt.rstrip()}\n{_CLOSE}\n"
        )
        return CorruptedSample(
            corrupted_source=wrapped,
        )
