"""Replace LF line endings with CRLF."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


class AddCrlf(InverseTransform):
    """Use Windows-style CRLF line endings."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_CRLF

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        # Normalize anything to \n first, then convert to \r\n.
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        return CorruptedSample(
            corrupted_source=normalized.replace("\n", "\r\n"),
        )
