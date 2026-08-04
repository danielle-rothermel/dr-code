"""Wrap source twice in code fences to simulate "Option A / Option B"."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.text_transforms import wrap_code_fence


class AddMultipleSolutions(Corruption):
    """Emit a "Option 1 / Option 2" message with two fenced blocks.

    The first block is the real source. The second block is a trivial
    placeholder. The recovery contract just requires the fence extractor
    to find both candidates — the validator picks the one that parses.
    """

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_MULTIPLE_SOLUTIONS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        alt = "def _alt_solution():\n    raise NotImplementedError\n"
        wrapped = (
            "Option 1:\n"
            f"{wrap_code_fence(source)}\n"
            "Option 2:\n"
            f"{wrap_code_fence(alt)}"
        )
        return CorruptedSample(
            corrupted_source=wrapped,
        )
