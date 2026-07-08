"""Wrap each line with a markdown prefix (blockquote / numbered / bullet)."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName, MarkdownWrapperMode
from dr_code.synthetic.inverse_transforms.base import InverseTransform

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


class AddMarkdownWrappers(InverseTransform):
    """Prefix each line with a markdown list/blockquote marker."""

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_MARKDOWN_WRAPPERS
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        mode = rng.choice(list(MarkdownWrapperMode))
        return CorruptedSample(
            corrupted_source=_apply_mode(source, mode),
            notes=f"mode={mode.value}",
        )
