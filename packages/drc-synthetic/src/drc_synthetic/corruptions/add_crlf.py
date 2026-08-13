from __future__ import annotations

import random
from typing import ClassVar

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName
from drc_synthetic.corruptions.base import Corruption
from dr_code.core.source.text_transforms import normalize_line_endings


class AddCrlf(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_CRLF
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=normalize_line_endings(source).replace(
                "\n", "\r\n"
            ),
        )
