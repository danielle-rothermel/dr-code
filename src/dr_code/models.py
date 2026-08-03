"""Shared frozen pydantic base for dr_code."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable, hashable base for definitions and persisted artifacts."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )
