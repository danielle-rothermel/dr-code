"""Corruption base class."""

from __future__ import annotations

import abc
import random
from typing import ClassVar, Generic, TypeVar, cast

from dr_code.core.models import FrozenModel
from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName


class CorruptionSettings(FrozenModel):
    """Base for per-corruption settings; each subclass declares its own.

    A corruption with no tunables uses this empty base directly. Settings
    are part of the persisted recipe coordinate: two recipes that differ
    only in settings are distinct semantic coordinates.
    """


SettingsT = TypeVar("SettingsT", bound=CorruptionSettings)


class Corruption(abc.ABC, Generic[SettingsT]):
    """Base class for synthetic corruptions.

    Subclasses must declare:
        NAME: the CorruptionName they implement.
        VERSION: the manual component version. In development mode it
            stays ``"0"``; bump it when the corruption's output for a
            given rng/text/settings changes, never for refactors.
        Settings: the concrete `CorruptionSettings` model this corruption
            reads. Corruptions with no tunables inherit the empty base.

    The `apply()` method takes the original ground-truth source and a
    `random.Random` instance (seeded by the dataset builder) and returns a
    `CorruptedSample`.

    Implementations must be deterministic given the rng and settings: same
    source + same settings + same rng state → same output.
    """

    NAME: ClassVar[CorruptionName]
    VERSION: ClassVar[str]
    Settings: ClassVar[type[CorruptionSettings]] = CorruptionSettings

    def __init__(self, settings: SettingsT | None = None) -> None:
        # Optional so corruptions with no tunables instantiate as
        # ``CorruptionCls()``; recipes always pass explicit validated
        # settings.
        self.settings: SettingsT = (
            settings
            if settings is not None
            else cast(SettingsT, self.Settings())
        )

    @abc.abstractmethod
    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        raise NotImplementedError
