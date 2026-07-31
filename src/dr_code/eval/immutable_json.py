"""Deeply immutable strict-JSON values with ordinary JSON serialization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import cast

from dr_serialize import validate_strict_json
from pydantic import JsonValue


class FrozenJsonDict(Mapping[str, object]):
    """An immutable, hashable JSON object with canonical key order."""

    _items: tuple[tuple[str, object], ...]
    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, object]) -> None:
        object.__setattr__(self, "_items", tuple(sorted(values.items())))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("frozen JSON objects do not support mutation")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("frozen JSON objects do not support mutation")

    def __getitem__(self, key: str) -> object:
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __setitem__(self, _key: str, _value: object) -> None:
        raise TypeError("frozen JSON objects do not support mutation")

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))


def freeze_json(value: object) -> object:
    """Validate and recursively freeze one strict JSON value."""

    return _freeze_validated_json(validate_strict_json(thaw_json(value)))


def _freeze_validated_json(value: JsonValue) -> object:
    if isinstance(value, dict):
        return FrozenJsonDict(
            {
                name: _freeze_validated_json(value[name])
                for name in sorted(value)
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_validated_json(item) for item in value)
    return value


def thaw_json(value: object) -> JsonValue:
    """Return a mutable, ordinary strict-JSON representation."""

    if isinstance(value, Mapping):
        return {str(name): thaw_json(child) for name, child in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(child) for child in value]
    return cast(JsonValue, validate_strict_json(value))


__all__ = ["FrozenJsonDict", "freeze_json", "thaw_json"]
