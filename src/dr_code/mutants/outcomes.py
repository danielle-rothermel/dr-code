"""Canonical schema-v2 execution outcomes for mutant wire and disk records."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class _OutcomeModel(BaseModel):
    """Strict immutable boundary shared by wire and persisted outcomes."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class ValueOutcome(_OutcomeModel):
    """A completed call whose return value is represented exactly."""

    kind: Literal["value"] = "value"
    value_repr: str


class ErrorOutcome(_OutcomeModel):
    """A completed call whose exception identity and arguments are preserved."""

    kind: Literal["error"] = "error"
    exception_type: str
    exception_args_repr: str


ExecutionOutcome: TypeAlias = Annotated[
    ValueOutcome | ErrorOutcome,
    Field(discriminator="kind"),
]


__all__ = (
    "ErrorOutcome",
    "ExecutionOutcome",
    "ValueOutcome",
)
