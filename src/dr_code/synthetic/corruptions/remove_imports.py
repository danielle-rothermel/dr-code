"""Remove existing `import` / `from ... import` statements."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.code_transforms import remove_top_level_imports
from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


class RemoveImports(Corruption):
    """Remove top-level import statements from the source."""

    NAME: ClassVar[CorruptionName] = CorruptionName.REMOVE_IMPORTS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        try:
            corrupted = remove_top_level_imports(source)
        except SyntaxError:
            corrupted = source
        return CorruptedSample(
            corrupted_source=corrupted,
        )
