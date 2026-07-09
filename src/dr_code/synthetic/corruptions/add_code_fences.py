"""Wrap source in a Markdown code fence (with or without language tag)."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import FenceLangTag, CorruptionName
from dr_code.synthetic.corruptions.base import Corruption
from dr_code.text_transforms import wrap_code_fence

#: Tags this transform may emit. Includes the untagged form.
_TAG_CHOICES: Final[tuple[FenceLangTag, ...]] = (
    FenceLangTag.PYTHON,
    FenceLangTag.PY,
    FenceLangTag.PYTHON3,
    FenceLangTag.NONE,
)


class AddCodeFences(Corruption):
    """Wrap the source in ```` ``` ```` with a randomly chosen tag (or none)."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_CODE_FENCES

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        tag = rng.choice(_TAG_CHOICES)
        return CorruptedSample(
            corrupted_source=wrap_code_fence(source, tag.value),
            notes=f"tag={tag.value!r}",
        )
