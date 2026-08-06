from __future__ import annotations

import ast
import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


def _annotate(source: str) -> str:
    try:
        ast.parse(source)
    except SyntaxError:
        return source
    return "_unused_annotated: int = 0\n" + source


class AddTypeAnnotations(Corruption):
    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_TYPE_ANNOTATIONS
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_annotate(source),
        )
