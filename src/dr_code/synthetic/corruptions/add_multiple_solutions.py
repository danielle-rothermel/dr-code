from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.core.source.text_transforms import wrap_code_fence


class AddMultipleSolutions(Corruption):
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
