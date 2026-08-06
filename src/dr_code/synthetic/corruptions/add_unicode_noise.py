from __future__ import annotations

import random
import unicodedata
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


class AddUnicodeNoise(Corruption):
    """Normalize the entire source to Unicode NFD."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_UNICODE_NOISE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=unicodedata.normalize("NFD", source),
        )
