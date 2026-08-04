"""Append trailing whitespace to each line."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption

_TRAILING_LEN_CHOICES: Final[tuple[int, ...]] = (1, 2, 3)


class AddTrailingWhitespace(Corruption):
    """Append a few spaces to each non-empty line."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_TRAILING_WHITESPACE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        n = rng.choice(_TRAILING_LEN_CHOICES)
        suffix = " " * n
        out_lines: list[str] = []
        for line in source.splitlines(keepends=True):
            if not line.strip():
                out_lines.append(line)
                continue
            if line.endswith("\n"):
                out_lines.append(line[:-1] + suffix + "\n")
            else:
                out_lines.append(line + suffix)
        return CorruptedSample(
            corrupted_source="".join(out_lines),
            notes=f"n={n}",
        )
