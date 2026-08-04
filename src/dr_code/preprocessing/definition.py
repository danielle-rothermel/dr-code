"""Frozen, serializable preprocessing definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import Field, SerializeAsAny, model_validator

from dr_code.models import FrozenModel
from dr_code.preprocessing.steps.base import StepSettings
from dr_code.trace import RESERVED_KEYS, WiringError
from dr_code.preprocessing.names import StepName


def _settings_payload(settings: object) -> object:
    """Reduce a settings model to a plain mapping so it is revalidated.

    Pydantic treats validating an instance of a subclass as a pass-through,
    which would let another component's settings through unchecked; a
    mapping always goes through full field validation.
    """

    if isinstance(settings, FrozenModel):
        return settings.model_dump()
    return settings


class StepSpec(FrozenModel):
    """One named step instance with its settings.

    ``instance_name`` becomes the trace key; renaming creates a new
    definition. ``settings`` is the registered step's concrete frozen model,
    resolved while the definition crosses this validation boundary.
    """

    instance_name: str
    step: StepName
    settings: SerializeAsAny[StepSettings] = Field(
        default_factory=StepSettings
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_settings_model(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if "step" not in data:
            # Let pydantic report the missing discriminator as a
            # ValidationError instead of raising KeyError out of band.
            return data
        step = StepName(data["step"])
        from dr_code.preprocessing.registry import REGISTRY

        settings_model = REGISTRY[step.value].Settings
        data["settings"] = settings_model.model_validate(
            _settings_payload(data.get("settings", {}))
        )
        return data


class PreprocessingDefinition(FrozenModel):
    """Ordered, named step instances and settings.

    Frozen and serializable. Fully describes the pipeline; no hidden defaults.
    """

    definition_id: str
    version: str
    steps: tuple[StepSpec, ...]
    __hash__ = None

    @model_validator(mode="after")
    def _validate_instance_names(self) -> Self:
        names = [spec.instance_name for spec in self.steps]
        # instance names must be unique and must not be reserved keys
        reserved = RESERVED_KEYS & set(names)
        if reserved:
            raise WiringError(
                "instance names must not be reserved keys: "
                + ", ".join(sorted(reserved))
            )
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        if duplicates:
            raise WiringError(
                "duplicate instance names: " + ", ".join(sorted(duplicates))
            )
        return self


__all__ = [
    "PreprocessingDefinition",
    "StepSpec",
]
