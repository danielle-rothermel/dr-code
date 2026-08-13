from __future__ import annotations

import random
from typing import ClassVar

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName
from drc_synthetic.corruptions.base import Corruption


class AddUnicodeNoise(Corruption):
    """Replace one ASCII character with its fullwidth look-alike."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_UNICODE_NOISE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        eligible = [
            index
            for index, character in enumerate(source)
            if "!" <= character <= "~"
        ]
        if not eligible:
            return CorruptedSample(corrupted_source=source)

        index = rng.choice(eligible)
        character = source[index]
        fullwidth = chr(ord(character) + 0xFEE0)
        return CorruptedSample(
            corrupted_source=source[:index] + fullwidth + source[index + 1 :],
        )
