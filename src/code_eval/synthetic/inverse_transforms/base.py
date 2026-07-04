"""Inverse transform base class."""

from __future__ import annotations

import random
from typing import ClassVar

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import InverseTransformName


class InverseTransform:
    """Base class for synthetic corruptions.

    Subclasses must declare:
        NAME: the InverseTransformName they implement.
        EXPECTED_RECOVERY_STEPS: frozenset[str] of step names the validator
            is expected to apply to undo this corruption.

    The `apply()` method takes the original ground-truth source and a
    `random.Random` instance (seeded by the dataset builder) and returns a
    `CorruptedSample`.

    Implementations must be deterministic given the rng: same source + same
    rng state → same output.
    """

    NAME: ClassVar[InverseTransformName]
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]]

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        raise NotImplementedError
