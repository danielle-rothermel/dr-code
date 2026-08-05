"""Replace ASCII quotes with Unicode "smart" quotes."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.core.source.text_transforms import SMART_QUOTES


def _replace_smart(source: str) -> str:
    """Replace ASCII quotes alternating left/right to mimic typographers."""
    out: list[str] = []
    open_single = True
    open_double = True
    for ch in source:
        if ch == "'":
            left, right = SMART_QUOTES["'"]
            out.append(left if open_single else right)
            open_single = not open_single
        elif ch == '"':
            left, right = SMART_QUOTES['"']
            out.append(left if open_double else right)
            open_double = not open_double
        else:
            out.append(ch)
    return "".join(out)


class AddSmartQuotes(Corruption):
    """Replace `'` and `"` with their Unicode 'smart' counterparts."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_SMART_QUOTES
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_replace_smart(source),
        )
