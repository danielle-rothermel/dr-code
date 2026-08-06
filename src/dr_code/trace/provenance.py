from __future__ import annotations

import math
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field, field_validator

from dr_code.core.models import FrozenModel


ComponentSettingValue: TypeAlias = (
    str | int | float | bool | None | tuple[str, ...]
)


class ComponentSetting(FrozenModel):
    name: str
    value: ComponentSettingValue

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


class StepCoordinate(FrozenModel):
    instance_name: str
    component: ComponentCoordinate


class PreprocessingDefinitionCoordinate(FrozenModel):
    definition_id: str
    version: str
    steps: tuple[StepCoordinate, ...]


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
