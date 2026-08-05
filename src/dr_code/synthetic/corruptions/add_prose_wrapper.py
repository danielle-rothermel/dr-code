"""Wrap source with prose intro and outro lines."""

from __future__ import annotations

import random
from typing import ClassVar, Final

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption

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


class AddProseWrapper(Corruption):
    """Prepend an intro and append an outro line."""

    NAME: ClassVar[CorruptionName] = CorruptionName.ADD_PROSE_WRAPPER
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        intro = rng.choice(_INTROS)
        outro = rng.choice(_OUTROS)
        parts = [intro, "", source.rstrip()]
        if outro:
            parts.extend(["", outro])
        wrapped = "\n".join(parts) + "\n"
        return CorruptedSample(
            corrupted_source=wrapped,
        )
