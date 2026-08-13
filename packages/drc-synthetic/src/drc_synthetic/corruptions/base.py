from __future__ import annotations

import abc
import random
from typing import ClassVar, Generic, TypeVar, cast

from dr_code.core.models import FrozenModel
from drc_synthetic.models import CorruptedSample
from drc_synthetic.names import CorruptionName


class CorruptionSettings(FrozenModel):
    pass


SettingsT = TypeVar("SettingsT", bound=CorruptionSettings)


class Corruption(abc.ABC, Generic[SettingsT]):
    NAME: ClassVar[CorruptionName]
    # In development mode, keep VERSION at "0". Afterward, bump it when output
    # changes for the same source, settings, and RNG state.
    VERSION: ClassVar[str]
    Settings: ClassVar[type[CorruptionSettings]] = CorruptionSettings

    def __init__(self, settings: SettingsT | None = None) -> None:
        self.settings: SettingsT = (
            settings
            if settings is not None
            else cast(SettingsT, self.Settings())
        )

    @abc.abstractmethod
    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        raise NotImplementedError
