from __future__ import annotations

import random
from typing import ClassVar

from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName
from drc_synthetic.corruptions.base import Corruption


class AddDeadCode(Corruption):
    """Prepend ``import os as _unused_module``."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_DEAD_CODE
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        injected = "import os as _unused_module  # noqa: F401\n"
        return CorruptedSample(
            corrupted_source=injected + source,
        )
