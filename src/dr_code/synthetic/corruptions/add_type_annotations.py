"""Add simple `: int = 0`-style annotations to module-level variables."""

from __future__ import annotations

import ast
import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


def _annotate(source: str) -> str:
    """Best-effort: tag a no-op `_x: int = 0` line near the top."""
    try:
        ast.parse(source)
    except SyntaxError:
        return source
    return "_unused_annotated: int = 0\n" + source


class AddTypeAnnotations(Corruption):
    """Inject an annotated module-level variable.

    Inverse of `dr_code.code_transforms.strip_type_annotations`.
    """

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_TYPE_ANNOTATIONS

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_annotate(source),
        )
