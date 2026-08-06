from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption

_INSERT_PROB: Final[float] = 0.25


class AddBlankLines(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_BLANK_LINES
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        out: list[str] = []
        for line in source.splitlines(keepends=True):
            out.append(line)
            if rng.random() < _INSERT_PROB:
                out.append("\n")
        return CorruptedSample(
            corrupted_source="".join(out),
        )
