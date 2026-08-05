"""Structured component and producer coordinates for typed traces."""

from __future__ import annotations

from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field

from dr_code.base import FrozenModel


ComponentSettingValue: TypeAlias = (
    str | int | float | bool | None | tuple[str, ...]
)


class ComponentSetting(FrozenModel):
    """One resolved scalar or ordered-string component setting."""

    name: str
    value: ComponentSettingValue


def coordinate_settings(
    settings: FrozenModel,
) -> tuple[ComponentSetting, ...]:
    """Project typed component settings into the bounded persisted shape.

    The projection dumps in ``python`` mode so ordered-string settings stay
    tuples, and rejects anything outside ``ComponentSettingValue`` so no
    component can widen the persisted coordinate by accident.
    """
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
    """One registered semantic component with resolved settings."""

    registered_name: str
    version: str
    settings: tuple[ComponentSetting, ...] = ()


class StepCoordinate(FrozenModel):
    """One named step instance in an ordered definition."""

    instance_name: str
    component: ComponentCoordinate


class PreprocessingDefinitionCoordinate(FrozenModel):
    """Complete semantic coordinate for a preprocessing definition."""

    definition_id: str
    version: str
    steps: tuple[StepCoordinate, ...]


class ExternalTraceProducer(FrozenModel):
    """A trace supplied from outside the registered component system."""

    kind: Literal["external"] = "external"


class PreprocessingTraceProducer(FrozenModel):
    """A trace produced by one completely resolved preprocessing definition."""

    kind: Literal["preprocessing"] = "preprocessing"
    definition: PreprocessingDefinitionCoordinate


class ExternalPreprocessingTraceProducer(FrozenModel):
    """A trace produced by an explicitly unregistered definition."""

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
