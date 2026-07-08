"""Wrap source in a Markdown code fence (with or without language tag)."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import FenceLangTag, InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform

_FENCE: Final[str] = "```"

#: Tags this transform may emit. Includes the untagged form.
_TAG_CHOICES: Final[tuple[FenceLangTag, ...]] = (
    FenceLangTag.PYTHON,
    FenceLangTag.PY,
    FenceLangTag.PYTHON3,
    FenceLangTag.NONE,
)


class AddCodeFences(InverseTransform):
    """Wrap the source in ```` ``` ```` with a randomly chosen tag (or none)."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_CODE_FENCES

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        tag = rng.choice(_TAG_CHOICES)
        opening = f"{_FENCE}{tag.value}" if tag.value else _FENCE
        wrapped = f"{opening}\n{source.rstrip()}\n{_FENCE}\n"
        return CorruptedSample(
            corrupted_source=wrapped,
            notes=f"tag={tag.value!r}",
        )
