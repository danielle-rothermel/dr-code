from __future__ import annotations

import random
from typing import ClassVar

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import FenceLangTag, CorruptionName
from drc_synthetic.corruptions.base import Corruption, CorruptionSettings
from dr_code.core.source.text_transforms import wrap_code_fence


class AddCodeFencesSettings(CorruptionSettings):
    language_tag: FenceLangTag = FenceLangTag.NONE


class AddCodeFences(Corruption[AddCodeFencesSettings]):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_CODE_FENCES
    VERSION: ClassVar[str] = "0"
    Settings: ClassVar[type[CorruptionSettings]] = AddCodeFencesSettings

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        tag = self.settings.language_tag
        return CorruptedSample(
            corrupted_source=wrap_code_fence(source, tag.value),
        )
