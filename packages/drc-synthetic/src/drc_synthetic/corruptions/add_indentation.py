from __future__ import annotations

import random
from typing import ClassVar, Final

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName
from drc_synthetic.corruptions.base import Corruption

_INDENT_CHOICES: Final[tuple[int, ...]] = (2, 4, 8)


class AddIndentation(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_INDENTATION
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        n = rng.choice(_INDENT_CHOICES)
        pad = " " * n
        indented = "".join(
            (pad + line if line.strip() else line)
            for line in source.splitlines(keepends=True)
        )
        return CorruptedSample(
            corrupted_source=indented,
        )
