"""Replace runs of leading spaces with tab characters."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.text_transforms import DEFAULT_TAB_WIDTH

_TAB: Final[str] = "\t"


def _spaces_to_tabs(source: str, tab_width: int) -> str:
    out_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        # Count leading spaces.
        i = 0
        while i < len(line) and line[i] == " ":
            i += 1
        if i == 0:
            out_lines.append(line)
            continue
        n_tabs = i // tab_width
        n_spaces = i % tab_width
        out_lines.append(_TAB * n_tabs + " " * n_spaces + line[i:])
    return "".join(out_lines)


class AddTabs(Corruption):
    """Convert leading spaces to tabs."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_TABS

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_spaces_to_tabs(source, DEFAULT_TAB_WIDTH),
        )
