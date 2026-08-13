from __future__ import annotations

import random
from typing import ClassVar, Final

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName
from drc_synthetic.corruptions.base import Corruption

_COMMENTS: Final[tuple[str, ...]] = (
    "# Step-by-step solution",
    "# This is the main logic",
    "# Note: handle edge cases here",
    "# TODO: optimize this later",
    "# Helper computation below",
)


class AddCommentsNoise(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_COMMENTS_NOISE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        out: list[str] = []
        for line in source.splitlines(keepends=True):
            if line.strip().startswith(
                ("def ", "class ", "if ", "for ", "while ", "return ")
            ):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{rng.choice(_COMMENTS)}\n")
            out.append(line)
        return CorruptedSample(
            corrupted_source="".join(out),
        )
