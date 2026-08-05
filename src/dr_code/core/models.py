"""Shared frozen Pydantic models for dr-code boundaries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable base for definitions and persisted artifacts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


def settings_payload(settings: object) -> object:
    """Reduce a settings model to a plain mapping so it is revalidated.

    Pydantic treats validating an instance of a subclass as a pass-through,
    which would let another component's settings through unchecked; a mapping
    always goes through full field validation.
    """

    if isinstance(settings, FrozenModel):
        return settings.model_dump()
    return settings


__all__ = [
    "FrozenModel",
    "settings_payload",
]
