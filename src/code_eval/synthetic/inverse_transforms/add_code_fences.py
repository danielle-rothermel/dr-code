"""Wrap source in a Markdown code fence (with or without language tag)."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import ExtractorName, FenceLangTag, InverseTransformName
from code_eval.synthetic.inverse_transforms.base import InverseTransform

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
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset({ExtractorName.FENCES.value})

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        tag = rng.choice(_TAG_CHOICES)
        opening = f"{_FENCE}{tag.value}" if tag.value else _FENCE
        wrapped = f"{opening}\n{source.rstrip()}\n{_FENCE}\n"
        return CorruptedSample(
            corrupted_source=wrapped,
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
            notes=f"tag={tag.value!r}",
        )
