from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import Field, field_validator, model_validator

from dr_code.core.models import FrozenModel


ComponentSettingValue: TypeAlias = (
    str | int | float | bool | None | tuple[str, ...]
)


class ComponentSetting(FrozenModel):
    name: str
    value: ComponentSettingValue

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComponentSetting):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return (
            self.name == other.name
            and type(self.value) is type(other.value)
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((type(self), self.name, type(self.value), self.value))

    @field_validator("value")
    @classmethod
    def reject_non_finite_float(
        cls, value: ComponentSettingValue
    ) -> ComponentSettingValue:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("component setting value must be finite")
        return value


def coordinate_settings(
    settings: FrozenModel,
) -> tuple[ComponentSetting, ...]:
    entries: list[ComponentSetting] = []
    for name, value in settings.model_dump(mode="python").items():
        if isinstance(value, tuple):
            if not all(isinstance(item, str) for item in value):
                raise TypeError(
                    f"unsupported persisted tuple setting for {name!r}"
                )
        if not isinstance(
            value, str | int | float | bool | type(None) | tuple
        ):
            raise TypeError(
                f"unsupported persisted setting shape for {name!r}: "
                f"{type(value).__name__}"
            )
        entries.append(ComponentSetting(name=name, value=value))
    return tuple(entries)


class ComponentCoordinate(FrozenModel):
    registered_name: str
    version: str
    settings: tuple[ComponentSetting, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_setting_names(self) -> Self:
        duplicates = _duplicate_names(
            setting.name for setting in self.settings
        )
        if duplicates:
            raise ValueError(
                "duplicate component setting names: " + ", ".join(duplicates)
            )
        return self


class StepCoordinate(FrozenModel):
    instance_name: str
    component: ComponentCoordinate


class PreprocessingDefinitionCoordinate(FrozenModel):
    definition_id: str
    version: str
    steps: tuple[StepCoordinate, ...]

    @model_validator(mode="after")
    def _validate_step_instance_names(self) -> Self:
        from dr_code.trace.trace import RESERVED_KEYS

        names = tuple(step.instance_name for step in self.steps)
        reserved = RESERVED_KEYS & set(names)
        if reserved:
            raise ValueError(
                "step instance names must not be reserved trace keys: "
                + ", ".join(sorted(reserved))
            )
        duplicates = _duplicate_names(names)
        if duplicates:
            raise ValueError(
                "duplicate step instance names: " + ", ".join(duplicates)
            )
        return self


def _duplicate_names(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return tuple(sorted(duplicates))


class ExternalTraceProducer(FrozenModel):
    kind: Literal["external"] = "external"


class PreprocessingTraceProducer(FrozenModel):
    kind: Literal["preprocessing"] = "preprocessing"
    definition: PreprocessingDefinitionCoordinate


class ExternalPreprocessingTraceProducer(FrozenModel):
    kind: Literal["external_preprocessing"] = "external_preprocessing"
    definition: PreprocessingDefinitionCoordinate


TraceProducer: TypeAlias = Annotated[
    ExternalTraceProducer
    | ExternalPreprocessingTraceProducer
    | PreprocessingTraceProducer,
    Field(discriminator="kind"),
]

EXTERNAL_PRODUCER: Final = ExternalTraceProducer()


__all__ = (
    "ComponentCoordinate",
    "ComponentSetting",
    "EXTERNAL_PRODUCER",
    "ExternalTraceProducer",
    "ExternalPreprocessingTraceProducer",
    "PreprocessingDefinitionCoordinate",
    "PreprocessingTraceProducer",
    "StepCoordinate",
    "TraceProducer",
    "coordinate_settings",
)
