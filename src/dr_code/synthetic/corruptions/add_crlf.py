"""Replace LF line endings with CRLF."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.core.source.text_transforms import normalize_line_endings


class AddCrlf(Corruption):
    """Use Windows-style CRLF line endings."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_CRLF
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        # Normalize anything to \n first, then convert to \r\n.
        return CorruptedSample(
            corrupted_source=normalize_line_endings(source).replace(
                "\n", "\r\n"
            ),
        )
