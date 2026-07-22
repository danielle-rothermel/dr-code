"""Variable declarations shared by every dr-code Definition.

A Definition is *variable-bearing*: it declares the unset Variables it
exposes (with optional allowed values / defaults), and materializes a
Config only from a complete assignment of every required Variable. This
module gives the minimal declaration + validation machinery; each
Definition declares its own Variables and constraints on top of it.
"""

from __future__ import annotations

from pydantic import JsonValue, model_validator

from dr_code.models import FrozenModel


class VariableError(ValueError):
    """A Config assignment did not satisfy its Definition's Variables."""


class VariableSpec(FrozenModel):
    """One declared Variable: a name, optional allowed values, default.

    ``allowed`` (when set) restricts the assignable values; ``default``
    (when set) makes the Variable optional in an assignment. A Variable
    with neither is required and unconstrained.
    """

    name: str
    allowed: tuple[JsonValue, ...] | None = None
    default: JsonValue | None = None
    has_default: bool = False

    @model_validator(mode="after")
    def _default_within_allowed(self) -> VariableSpec:
        if (
            self.has_default
            and self.allowed is not None
            and self.default not in self.allowed
        ):
            raise VariableError(
                f"default for {self.name!r} is not in its allowed values"
            )
        return self


def resolve_assignment(
    specs: tuple[VariableSpec, ...],
    assignment: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate ``assignment`` against ``specs`` and fill defaults.

    Raises :class:`VariableError` on unknown, missing, or out-of-range
    Variables. Returns the complete resolved assignment (declared order).
    """

    spec_by_name = {spec.name: spec for spec in specs}
    unknown = set(assignment) - set(spec_by_name)
    if unknown:
        raise VariableError(
            "unknown variables: " + ", ".join(sorted(unknown))
        )

    resolved: dict[str, JsonValue] = {}
    for spec in specs:
        if spec.name in assignment:
            value = assignment[spec.name]
        elif spec.has_default:
            value = spec.default
        else:
            raise VariableError(f"variable {spec.name!r} is unassigned")
        if spec.allowed is not None and value not in spec.allowed:
            raise VariableError(
                f"value for {spec.name!r} is not an allowed value"
            )
        resolved[spec.name] = value
    return resolved


__all__ = [
    "VariableError",
    "VariableSpec",
    "resolve_assignment",
]
