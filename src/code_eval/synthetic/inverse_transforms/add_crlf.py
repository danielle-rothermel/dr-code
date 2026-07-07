"""Replace LF line endings with CRLF."""

from __future__ import annotations

import random
from typing import ClassVar

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import ExtractorName, InverseTransformName
from code_eval.synthetic.inverse_transforms.base import InverseTransform


class AddCrlf(InverseTransform):
    """Use Windows-style CRLF line endings."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_CRLF
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {ExtractorName.TEXT_NORMALIZE.value}
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        # Normalize anything to \n first, then convert to \r\n.
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        return CorruptedSample(
            corrupted_source=normalized.replace("\n", "\r\n"),
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
        )
