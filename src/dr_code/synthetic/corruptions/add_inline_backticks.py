"""Wrap source in a single-backtick inline span."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


class AddInlineBackticks(Corruption):
    """Wrap the entire source in a single-backtick inline span."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_INLINE_BACKTICKS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=f"`{source.rstrip()}`",
        )
