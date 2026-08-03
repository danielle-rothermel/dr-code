"""Corruption base class."""

from __future__ import annotations

import abc
import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName


class Corruption(abc.ABC):
    """Base class for synthetic corruptions.

    Subclasses must declare:
        NAME: the CorruptionName they implement.

    The `apply()` method takes the original ground-truth source and a
    `random.Random` instance (seeded by the dataset builder) and returns a
    `CorruptedSample`.

    Implementations must be deterministic given the rng: same source + same
    rng state → same output.
    """

    NAME: ClassVar[CorruptionName]
    VERSION: ClassVar[str]

    @abc.abstractmethod
    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        raise NotImplementedError
