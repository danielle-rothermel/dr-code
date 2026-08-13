from __future__ import annotations

import random
from typing import ClassVar

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName
from drc_synthetic.corruptions.base import Corruption


class AddInlineBackticks(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_INLINE_BACKTICKS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=f"`{source.rstrip()}`",
        )
