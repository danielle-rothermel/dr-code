"""Typed variable declarations and substitution for evaluation definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self, cast

from dr_serialize import validate_strict_json
from pydantic import (
    JsonValue,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from dr_code.models import FrozenModel


class VariableError(ValueError):
    """A config assignment does not satisfy its definition."""


type NormalizedJson = str | int | float | bool | None | JsonArray | JsonObject


class JsonObject(FrozenModel):
    """One hashable JSON object stored as name/value entries in name order.

    Definition settings and variable values live inside frozen models, so the
    JSON they carry must be hashable: objects become entry tuples and arrays
    become ``JsonArray``. A JSON object is an unordered mapping, so entries are
    stored sorted by name and two objects are equal exactly when their entries
    match pairwise -- key order in the source object carries no identity.
    """

    entries: tuple[tuple[str, NormalizedJson], ...] = ()

    def as_dict(self) -> dict[str, NormalizedJson]:
        return dict(self.entries)

    @model_validator(mode="before")
    @classmethod
    def accept_a_plain_json_object(cls, value: object) -> object:
        """Accept either stored entries or the plain object they represent."""

        if isinstance(value, Mapping):
            mapping = cast(Mapping[str, object], value)
            if set(mapping) != {"entries"}:
                return {
                    "entries": tuple(
                        sorted(
                            (str(name), normalize_json(child))
                            for name, child in mapping.items()
                        )
                    )
                }
        return value

    @model_serializer
    def serialize_as_a_plain_json_object(self) -> dict[str, JsonValue]:
        """Serialize as the ordinary JSON object this value represents."""

        return {name: denormalize_json(child) for name, child in self.entries}

    def __eq__(self, other: object) -> bool:
        """Compare entries by exact recursive type, keeping 1 and True apart.

        Pydantic's generated equality compares entries with plain ``==``,
        under which ``1 == True`` and ``1 == 1.0``. JSON scalars of different
        types are different values here, so equality checks types too.
        """

        if not isinstance(other, JsonObject):
            return NotImplemented
        return _json_values_equal(self, other)

    def __hash__(self) -> int:
        return hash(self.entries)


class JsonArray(FrozenModel):
    """One hashable JSON array stored as an ordered tuple of items.

    Arrays are ordered, so item order carries identity. Items compare by exact
    recursive type through the same mechanism objects use, so ``[1]``,
    ``[1.0]``, and ``[True]`` are three distinct arrays.
    """

    items: tuple[NormalizedJson, ...] = ()

    def as_list(self) -> list[NormalizedJson]:
        return list(self.items)

    @model_validator(mode="before")
    @classmethod
    def accept_a_plain_json_array(cls, value: object) -> object:
        """Accept either stored items or the plain array they represent."""

        if isinstance(value, (list, tuple)):
            sequence = cast(Sequence[object], value)
            return {"items": tuple(normalize_json(item) for item in sequence)}
        return value

    @model_serializer
    def serialize_as_a_plain_json_array(self) -> list[JsonValue]:
        """Serialize as the ordinary JSON array this value represents."""

        return [denormalize_json(item) for item in self.items]

    def __eq__(self, other: object) -> bool:
        """Compare items by exact recursive type, keeping 1 and True apart."""

        if not isinstance(other, JsonArray):
            return NotImplemented
        return _json_values_equal(self, other)

    def __hash__(self) -> int:
        return hash(self.items)


def normalize_json(value: object) -> NormalizedJson:
    """Validate one strict JSON value into its hashable normalized form."""

    return _normalize_validated_json(
        cast(JsonValue, validate_strict_json(denormalize_json(value)))
    )


def _normalize_validated_json(value: JsonValue) -> NormalizedJson:
    if isinstance(value, dict):
        return JsonObject(
            entries=tuple(
                sorted(
                    (name, _normalize_validated_json(child))
                    for name, child in value.items()
                )
            )
        )
    if isinstance(value, list):
        return JsonArray(
            items=tuple(_normalize_validated_json(item) for item in value)
        )
    return value


def denormalize_json(value: object) -> JsonValue:
    """Return the ordinary mutable JSON form of a normalized value."""

    if isinstance(value, JsonObject):
        return {name: denormalize_json(child) for name, child in value.entries}
    if isinstance(value, JsonArray):
        return [denormalize_json(item) for item in value.items]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {
            str(name): denormalize_json(child)
            for name, child in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return [denormalize_json(child) for child in items]
    return cast(JsonValue, validate_strict_json(value))


class VariableReference(FrozenModel):
    """A typed reference embedded in a definition setting."""

    variable: str

    @field_validator("variable")
    @classmethod
    def reject_empty_name(cls, value: str) -> str:
        if not value:
            raise VariableError("variable reference name must not be empty")
        return value


def _json_values_equal(left: NormalizedJson, right: NormalizedJson) -> bool:
    """Compare two normalized JSON values by exact recursive type and value.

    Every level compares concrete types before values, so ``1``, ``1.0``, and
    ``True`` stay distinct however deeply they are nested. Plain ``==`` is not
    enough on its own: Python compares ``1 == True`` as true, and pydantic
    model equality inherits that inside nested entries.
    """

    if type(left) is not type(right):
        return False
    if isinstance(left, JsonObject):
        assert isinstance(right, JsonObject)
        return len(left.entries) == len(right.entries) and all(
            left_name == right_name
            and _json_values_equal(left_child, right_child)
            for (left_name, left_child), (right_name, right_child) in zip(
                left.entries, right.entries, strict=True
            )
        )
    if isinstance(left, JsonArray):
        assert isinstance(right, JsonArray)
        return len(left.items) == len(right.items) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(
                left.items, right.items, strict=True
            )
        )
    return left == right


def _is_allowed(
    value: NormalizedJson,
    allowed: tuple[NormalizedJson, ...],
) -> bool:
    return any(_json_values_equal(value, candidate) for candidate in allowed)


class VariableSpec(FrozenModel):
    """One serialized variable declaration."""

    name: str
    allowed: tuple[NormalizedJson, ...] | None = None
    default: NormalizedJson = None
    has_default: bool = False

    @field_validator("allowed", mode="before")
    @classmethod
    def normalize_allowed_values(cls, value: object) -> object:
        if value is None:
            return value
        return tuple(
            normalize_json(item)
            for item in cast(tuple[object, ...] | list[object], value)
        )

    @field_validator("default", mode="before")
    @classmethod
    def normalize_default_json(cls, value: object) -> object:
        return None if value is None else normalize_json(value)

    @field_serializer("allowed")
    def serialize_allowed(
        self, value: tuple[NormalizedJson, ...] | None
    ) -> object:
        if value is None:
            return None
        return [denormalize_json(item) for item in value]

    @field_serializer("default")
    def serialize_default(self, value: NormalizedJson) -> JsonValue:
        return denormalize_json(value)

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
    assignment: dict[str, NormalizedJson],
) -> dict[str, NormalizedJson]:
    """Validate and complete an assignment in declaration order."""

    spec_by_name = {spec.name: spec for spec in specs}
    if len(spec_by_name) != len(specs):
        raise VariableError("variable names must be unique")
    unknown = set(assignment) - set(spec_by_name)
    if unknown:
        raise VariableError("unknown variables: " + ", ".join(sorted(unknown)))

    resolved: dict[str, NormalizedJson] = {}
    for spec in specs:
        if spec.name in assignment:
            value = normalize_json(assignment[spec.name])
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
        elif isinstance(item, JsonObject):
            for _name, child in item.entries:
                visit(child)
        elif isinstance(item, JsonArray):
            for child in item.items:
                visit(child)
        elif isinstance(item, Mapping):
            mapping = cast(Mapping[str, object], item)
            for child in mapping.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in cast(Sequence[object], item):
                visit(child)

    visit(value)
    return tuple(references)


def substitute_variables(
    value: object,
    assignment: dict[str, NormalizedJson],
) -> JsonValue:
    """Resolve typed references recursively while preserving exact JSON types."""

    if isinstance(value, VariableReference):
        try:
            return denormalize_json(assignment[value.variable])
        except KeyError as exc:  # defensive for direct helper callers
            raise VariableError(
                f"variable {value.variable!r} is unassigned"
            ) from exc
    if isinstance(value, JsonObject):
        return {
            name: substitute_variables(child, assignment)
            for name, child in value.entries
        }
    if isinstance(value, JsonArray):
        return [substitute_variables(item, assignment) for item in value.items]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {
            str(name): substitute_variables(child, assignment)
            for name, child in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return [substitute_variables(child, assignment) for child in items]
    return cast(JsonValue, validate_strict_json(value))


__all__ = [
    "JsonArray",
    "JsonObject",
    "NormalizedJson",
    "VariableError",
    "VariableReference",
    "VariableSpec",
    "denormalize_json",
    "normalize_json",
    "resolve_assignment",
    "substitute_variables",
    "variable_references",
]
