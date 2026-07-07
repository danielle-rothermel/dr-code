"""Wrap source with prose intro and outro lines."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import ExtractorName, InverseTransformName
from code_eval.synthetic.inverse_transforms.base import InverseTransform

_INTROS: Final[tuple[str, ...]] = (
    "Here's the solution:",
    "Here is the code:",
    "Sure! Here's my answer:",
    "Below is the program:",
    "I'd write it like this:",
)

_OUTROS: Final[tuple[str, ...]] = (
    "Let me know if you have any questions!",
    "Hope this helps.",
    "Feel free to ask for clarification.",
    "",
)


class AddProseWrapper(InverseTransform):
    """Prepend an intro and append an outro line."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.ADD_PROSE_WRAPPER
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {ExtractorName.PROSE_PATTERNS.value, ExtractorName.FENCES.value}
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        intro = rng.choice(_INTROS)
        outro = rng.choice(_OUTROS)
        parts = [intro, "", source.rstrip()]
        if outro:
            parts.extend(["", outro])
        wrapped = "\n".join(parts) + "\n"
        return CorruptedSample(
            corrupted_source=wrapped,
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
            notes=f"intro={intro!r} outro={outro!r}",
        )
