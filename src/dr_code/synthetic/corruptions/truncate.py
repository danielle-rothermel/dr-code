"""Truncate the source mid-function / mid-line / mid-string."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName, TruncationMode
from dr_code.synthetic.corruptions.base import Corruption


def _truncate(source: str, mode: TruncationMode, rng: random.Random) -> str:
    lines = source.splitlines(keepends=True)
    if len(lines) <= 2:
        # too short to truncate meaningfully — drop the last character.
        return source[:-1] if source else source

    if mode is TruncationMode.MID_FUNCTION:
        # Drop the last 1-3 lines of the source.
        n_drop = min(rng.randint(1, 3), len(lines) - 2)
        return "".join(lines[:-n_drop])

    if mode is TruncationMode.MID_LINE:
        # Find a non-empty line in the last third and chop it.
        candidates = [
            i
            for i in range(int(len(lines) * 2 / 3), len(lines))
            if lines[i].strip()
        ]
        if not candidates:
            return "".join(lines[:-1])
        idx = rng.choice(candidates)
        line = lines[idx]
        cut = max(1, len(line.rstrip()) // 2)
        return "".join(lines[:idx]) + line[:cut]

    # mid_string: open a string and don't close it.
    return "".join(lines) + '\nmessage = "this is an unterminated string'


class Truncate(Corruption):
    """Cut the source short to simulate token-limit truncation."""

    NAME: ClassVar[CorruptionName] = CorruptionName.TRUNCATE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        mode = rng.choice(list(TruncationMode))
        return CorruptedSample(
            corrupted_source=_truncate(source, mode, rng),
        )
