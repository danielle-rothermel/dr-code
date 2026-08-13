from __future__ import annotations

import random
from typing import ClassVar, Final

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName, MarkdownWrapperMode
from drc_synthetic.corruptions.base import Corruption

_BLOCKQUOTE_PREFIX: Final[str] = "> "
_BULLET_PREFIX: Final[str] = "- "


def _apply_mode(source: str, mode: MarkdownWrapperMode) -> str:
    out: list[str] = []
    n = 1
    for line in source.splitlines(keepends=True):
        if not line.strip():
            out.append(line)
            continue
        if mode is MarkdownWrapperMode.BLOCKQUOTE:
            out.append(_BLOCKQUOTE_PREFIX + line)
        elif mode is MarkdownWrapperMode.NUMBERED_LIST:
            out.append(f"{n}. {line}")
            n += 1
        elif mode is MarkdownWrapperMode.BULLET_LIST:
            out.append(_BULLET_PREFIX + line)
    return "".join(out)


class AddMarkdownWrappers(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_MARKDOWN_WRAPPERS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        mode = rng.choice(list(MarkdownWrapperMode))
        return CorruptedSample(
            corrupted_source=_apply_mode(source, mode),
        )
