"""Typed variable declarations and substitution for evaluation definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from dr_serialize import validate_strict_json
from pydantic import (
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from dr_code.eval.immutable_json import freeze_json, thaw_json
from dr_code.models import FrozenModel


class VariableError(ValueError):
    """A config assignment does not satisfy its definition."""


class VariableReference(FrozenModel):
    """A typed reference embedded in a definition setting."""

    variable: str

    @field_validator("variable")
    @classmethod
    def reject_empty_name(cls, value: str) -> str:
        if not value:
            raise VariableError("variable reference name must not be empty")
        return value


def _json_values_equal(left: JsonValue, right: JsonValue) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        left_mapping = cast(Mapping[str, JsonValue], left)
        right_mapping = cast(Mapping[str, JsonValue], right)
        return left_mapping.keys() == right_mapping.keys() and all(
            _json_values_equal(left_mapping[key], right_mapping[key])
            for key in left_mapping
        )
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _is_allowed(
    value: JsonValue,
    allowed: tuple[JsonValue, ...],
) -> bool:
    return any(_json_values_equal(value, candidate) for candidate in allowed)


class VariableSpec(FrozenModel):
    """One serialized variable declaration."""

    name: str
    allowed: tuple[JsonValue, ...] | None = None
    default: JsonValue | None = None
    has_default: bool = False

    @field_validator("allowed", mode="before")
    @classmethod
    def validate_allowed_values(cls, value: object) -> object:
        if value is None:
            return value
        return tuple(
            validate_strict_json(item)
            for item in cast(tuple[object, ...] | list[object], value)
        )

    @field_validator("allowed", mode="after")
    @classmethod
    def freeze_allowed_values(
        cls, value: tuple[JsonValue, ...] | None
    ) -> tuple[JsonValue, ...] | None:
        if value is None:
            return None
        return tuple(cast(JsonValue, freeze_json(item)) for item in value)

    @field_validator("default", mode="before")
    @classmethod
    def validate_default_json(cls, value: object) -> object:
        return None if value is None else validate_strict_json(value)

    @field_validator("default", mode="after")
    @classmethod
    def freeze_default_json(cls, value: JsonValue) -> JsonValue:
        return cast(JsonValue, freeze_json(value))

    @field_serializer("allowed")
    def serialize_allowed(self, value: tuple[JsonValue, ...] | None) -> object:
        if value is None:
            return None
        return [thaw_json(item) for item in value]

    @field_serializer("default")
    def serialize_default(self, value: JsonValue) -> JsonValue:
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_default(self) -> Self:
        if not self.name:
            raise VariableError("variable name must not be empty")
        if (
            self.has_default
            and self.allowed is not None
            and not _is_allowed(self.default, self.allowed)
        ):
            raise VariableError(
                f"default for {self.name!r} is not in its allowed values"
            )
        return self


def resolve_assignment(
    specs: tuple[VariableSpec, ...],
    assignment: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Validate and complete an assignment in declaration order."""

    spec_by_name = {spec.name: spec for spec in specs}
    if len(spec_by_name) != len(specs):
        raise VariableError("variable names must be unique")
    unknown = set(assignment) - set(spec_by_name)
    if unknown:
        raise VariableError("unknown variables: " + ", ".join(sorted(unknown)))

    resolved: dict[str, JsonValue] = {}
    for spec in specs:
        if spec.name in assignment:
            value = cast(JsonValue, freeze_json(assignment[spec.name]))
        elif spec.has_default:
            value = spec.default
        else:
            raise VariableError(f"variable {spec.name!r} is unassigned")
        if spec.allowed is not None and not _is_allowed(value, spec.allowed):
            raise VariableError(
                f"value for {spec.name!r} is not an allowed value"
            )
        resolved[spec.name] = value
    return resolved


def variable_references(value: object) -> tuple[str, ...]:
    """Return variable names referenced recursively in one template value."""

    references: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, VariableReference):
            references.append(item.variable)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(references)


def substitute_variables(
    value: object,
    assignment: dict[str, JsonValue],
) -> JsonValue:
    """Resolve typed references recursively while preserving exact JSON types."""

    if isinstance(value, VariableReference):
        try:
            return assignment[value.variable]
        except KeyError as exc:  # defensive for direct helper callers
            raise VariableError(
                f"variable {value.variable!r} is unassigned"
            ) from exc
    if isinstance(value, Mapping):
        return {
            str(name): substitute_variables(child, assignment)
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [substitute_variables(child, assignment) for child in value]
    return validate_strict_json(value)


__all__ = [
    "VariableError",
    "VariableReference",
    "VariableSpec",
    "resolve_assignment",
    "substitute_variables",
    "variable_references",
]
