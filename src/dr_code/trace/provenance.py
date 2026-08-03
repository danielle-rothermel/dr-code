"""Structured component and producer coordinates for typed traces."""

from __future__ import annotations

from typing import Annotated, Final, Literal, TypeAlias

from pydantic import Field

from dr_code.models import FrozenModel


ComponentSettingValue: TypeAlias = (
    str | int | float | bool | None | tuple[str, ...]
)


class ComponentSetting(FrozenModel):
    """One resolved scalar or ordered-string component setting."""

    name: str
    value: ComponentSettingValue


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
)
