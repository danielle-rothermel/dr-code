"""Apply benign Unicode-form corruption (NFC -> NFD)."""

from __future__ import annotations

import random
import unicodedata
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


class AddUnicodeNoise(InverseTransform):
    """Convert to NFD (decomposed) form.

    HumanEval sources are mostly ASCII; adding non-breaking spaces or
    decomposing the few accented characters that may appear is enough to
    exercise the text-normalize NFC pass without changing semantics.
    """

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_UNICODE_NOISE
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        # Decompose to NFD; for pure-ASCII inputs this is a no-op, but the
        # transform is still legal — the text-normalize step's NFC pass is
        # idempotent on ASCII, so the recovery contract holds trivially.
        return CorruptedSample(
            corrupted_source=unicodedata.normalize("NFD", source),
        )
