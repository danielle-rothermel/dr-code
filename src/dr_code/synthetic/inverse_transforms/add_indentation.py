"""Prepend uniform leading whitespace to every line.

Simulates chat clients that render code with a fixed left margin.
"""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform

_INDENT_CHOICES: Final[tuple[int, ...]] = (2, 4, 8)


class AddIndentation(InverseTransform):
    """Indent every line uniformly by N spaces."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_INDENTATION

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        n = rng.choice(_INDENT_CHOICES)
        pad = " " * n
        indented = "".join(
            (pad + line if line.strip() else line)
            for line in source.splitlines(keepends=True)
        )
        return CorruptedSample(
            corrupted_source=indented,
            notes=f"n={n}",
        )
