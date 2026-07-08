"""Replace ASCII quotes with Unicode "smart" quotes."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform

#: Mapping from ASCII to Unicode smart quote (single, double).
_SMART_QUOTE_MAP: Final[dict[str, tuple[str, str]]] = {
    "'": ("\u2018", "\u2019"),  # left, right single
    '"': ("\u201c", "\u201d"),  # left, right double
}


def _replace_smart(source: str) -> str:
    """Replace ASCII quotes alternating left/right to mimic typographers."""
    out: list[str] = []
    open_single = True
    open_double = True
    for ch in source:
        if ch == "'":
            left, right = _SMART_QUOTE_MAP["'"]
            out.append(left if open_single else right)
            open_single = not open_single
        elif ch == '"':
            left, right = _SMART_QUOTE_MAP['"']
            out.append(left if open_double else right)
            open_double = not open_double
        else:
            out.append(ch)
    return "".join(out)


class AddSmartQuotes(InverseTransform):
    """Replace `'` and `"` with their Unicode 'smart' counterparts."""

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_SMART_QUOTES
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_replace_smart(source),
        )
