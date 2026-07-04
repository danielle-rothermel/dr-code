"""Insert extra blank lines throughout the source."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import ExtractorName, InverseTransformName
from code_eval.synthetic.inverse_transforms.base import InverseTransform

#: Probability that each line gets a blank inserted after it.
_INSERT_PROB: Final[float] = 0.25


class AddBlankLines(InverseTransform):
    """Sprinkle extra blank lines between lines."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_BLANK_LINES
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {ExtractorName.TEXT_NORMALIZE.value}
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        out: list[str] = []
        for line in source.splitlines(keepends=True):
            out.append(line)
            if rng.random() < _INSERT_PROB:
                out.append("\n")
        return CorruptedSample(
            corrupted_source="".join(out),
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
        )
