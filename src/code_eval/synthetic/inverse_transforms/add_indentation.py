"""Prepend uniform leading whitespace to every line.

Simulates chat clients that render code with a fixed left margin.
"""

from __future__ import annotations

import random
from typing import ClassVar, Final

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import InverseTransformName, RepairName
from code_eval.synthetic.inverse_transforms.base import InverseTransform

_INDENT_CHOICES: Final[tuple[int, ...]] = (2, 4, 8)


class AddIndentation(InverseTransform):
    """Indent every line uniformly by N spaces."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_INDENTATION
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset({RepairName.DEDENT.value})

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        n = rng.choice(_INDENT_CHOICES)
        pad = " " * n
        indented = "".join(
            (pad + line if line.strip() else line) for line in source.splitlines(keepends=True)
        )
        return CorruptedSample(
            corrupted_source=indented,
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
            notes=f"n={n}",
        )
