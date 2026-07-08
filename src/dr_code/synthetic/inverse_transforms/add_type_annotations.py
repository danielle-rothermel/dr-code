"""Add simple `: int = 0`-style annotations to module-level variables."""

from __future__ import annotations

import ast
import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


def _annotate(source: str) -> str:
    """Best-effort: tag a no-op `_x: int = 0` line near the top."""
    try:
        ast.parse(source)
    except SyntaxError:
        return source
    return "_unused_annotated: int = 0\n" + source


class AddTypeAnnotations(InverseTransform):
    """Inject an annotated module-level variable to be stripped by `annotation_strip`."""

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_TYPE_ANNOTATIONS
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_annotate(source),
        )
