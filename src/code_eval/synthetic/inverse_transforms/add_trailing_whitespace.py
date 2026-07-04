"""Append trailing whitespace to each line."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import ExtractorName, InverseTransformName
from code_eval.synthetic.inverse_transforms.base import InverseTransform

_TRAILING_LEN_CHOICES: Final[tuple[int, ...]] = (1, 2, 3)


class AddTrailingWhitespace(InverseTransform):
    """Append a few spaces to each non-empty line."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_TRAILING_WHITESPACE
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {ExtractorName.TEXT_NORMALIZE.value}
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        n = rng.choice(_TRAILING_LEN_CHOICES)
        suffix = " " * n
        out_lines: list[str] = []
        for line in source.splitlines(keepends=True):
            if not line.strip():
                out_lines.append(line)
                continue
            if line.endswith("\n"):
                out_lines.append(line[:-1] + suffix + "\n")
            else:
                out_lines.append(line + suffix)
        return CorruptedSample(
            corrupted_source="".join(out_lines),
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
            notes=f"n={n}",
        )
