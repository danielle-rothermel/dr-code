"""Frozen, serializable, hashable preprocessing definitions."""

from __future__ import annotations

import json

from pydantic import JsonValue, model_validator

from dr_code.models import FrozenModel
from dr_code.trace import RESERVED_KEYS, WiringError, stable_hash
from dr_code.preprocessing.names import StepName


class StepSpec(FrozenModel):
    """One named step instance with its settings.

    ``instance_name`` becomes the trace key; renaming creates a new
    definition. ``settings`` stays a ``dict[str, JsonValue]``, validated
    against the step's ``Settings`` model at bind time.
    """

    instance_name: str
    step: StepName
    settings: dict[str, JsonValue] = {}


class PreprocessingDefinition(FrozenModel):
    """Ordered, named step instances and settings.

    Frozen, serializable, hashable. Fully describes the pipeline; no
    hidden defaults.
    """

    definition_id: str
    version: str
    steps: tuple[StepSpec, ...]

    def __hash__(self) -> int:
        # ``settings`` holds a ``dict[str, JsonValue]`` (unhashable), so
        # pydantic's generated hash cannot apply. Hash the sorted JSON
        # form instead — consistent with ``preprocessing_definition_hash``
        # and ``__eq__`` (equal definitions hash equal).
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hash(blob)

    @model_validator(mode="after")
    def _validate_instance_names(self) -> PreprocessingDefinition:
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


def preprocessing_definition_hash(
    definition: PreprocessingDefinition,
) -> str:
    """``trace.identity.stable_hash`` — the deterministic identity for sweeps."""
    return stable_hash(definition)


__all__ = [
    "PreprocessingDefinition",
    "StepSpec",
    "preprocessing_definition_hash",
]
