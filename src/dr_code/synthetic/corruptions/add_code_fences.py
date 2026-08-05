"""Wrap source in a Markdown code fence with an explicitly chosen tag."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import FenceLangTag, CorruptionName
from dr_code.synthetic.corruptions.base import Corruption, CorruptionSettings
from dr_code.core.source.text_transforms import wrap_code_fence


class AddCodeFencesSettings(CorruptionSettings):
    """Which language tag the emitted fence carries."""

    language_tag: FenceLangTag = FenceLangTag.NONE


class AddCodeFences(Corruption[AddCodeFencesSettings]):
    """Wrap the source in ```` ``` ```` with the configured tag (or none)."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_CODE_FENCES
    VERSION: ClassVar[str] = "0"
    Settings: ClassVar[type[CorruptionSettings]] = AddCodeFencesSettings

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        tag = self.settings.language_tag
        return CorruptedSample(
            corrupted_source=wrap_code_fence(source, tag.value),
        )
