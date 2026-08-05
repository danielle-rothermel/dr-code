from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


def settings_payload(settings: object) -> object:
    """Dump payloads to prevent Pydantic subclass-instance pass-through."""

    if isinstance(settings, FrozenModel):
        return settings.model_dump()
    return settings


__all__ = [
    "FrozenModel",
    "settings_payload",
]
