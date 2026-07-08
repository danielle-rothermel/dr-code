"""Inject prose-y comments throughout the source."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform

_COMMENTS: Final[tuple[str, ...]] = (
    "# Step-by-step solution",
    "# This is the main logic",
    "# Note: handle edge cases here",
    "# TODO: optimize this later",
    "# Helper computation below",
)


class AddCommentsNoise(InverseTransform):
    """Inject prose comments. Recovery is by stripping comments during L1."""

    NAME: ClassVar[InverseTransformName] = (
        InverseTransformName.ADD_COMMENTS_NOISE
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        out: list[str] = []
        for line in source.splitlines(keepends=True):
            if line.strip().startswith(
                ("def ", "class ", "if ", "for ", "while ", "return ")
            ):
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{rng.choice(_COMMENTS)}\n")
            out.append(line)
        return CorruptedSample(
            corrupted_source="".join(out),
        )
