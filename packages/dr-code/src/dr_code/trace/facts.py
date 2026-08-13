from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeAlias

from pydantic import JsonValue

JsonFactValue: TypeAlias = JsonValue

StepFacts: TypeAlias = Mapping[str, Mapping[str, JsonFactValue]]


class FactError(ValueError):
    pass


def validate_step_facts(
    step_facts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, JsonFactValue]]:
    """Copy into finite, acyclic JSON with string keys and plain built-ins."""

    if not isinstance(step_facts, Mapping):
        raise FactError(
            f"step facts must be a mapping: {type(step_facts).__name__}"
        )
    validated: dict[str, dict[str, JsonFactValue]] = {}
    for instance_name, facts in step_facts.items():
        if not isinstance(instance_name, str):
            raise FactError(
                "step fact instance names must be strings: "
                f"{type(instance_name).__name__}"
            )
        if not isinstance(facts, Mapping):
            raise FactError(
                f"step facts for {instance_name!r} must be a mapping: "
                f"{type(facts).__name__}"
            )
        entries: dict[str, JsonFactValue] = {}
        for key, value in facts.items():
            if not isinstance(key, str):
                raise FactError(
                    f"step fact keys for {instance_name!r} must be strings: "
                    f"{type(key).__name__}"
                )
            entries[key] = _validate_value(
                value, path=f"{instance_name}.{key}", seen=()
            )
        validated[instance_name] = entries
    return validated


def _validate_value(
    value: object,
    *,
    path: str,
    seen: tuple[int, ...],
) -> JsonFactValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value if type(value) is str else str(value)
    if isinstance(value, int):
        return value if type(value) is int else int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FactError(f"step fact {path} is a non-finite float: {value}")
        return value if type(value) is float else float(value)
    if isinstance(value, Mapping):
        _reject_cycle(value, path=path, seen=seen)
        nested = (*seen, id(value))
        mapping: dict[str, JsonFactValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FactError(
                    f"step fact {path} has a non-string key: "
                    f"{type(key).__name__}"
                )
            mapping[key] = _validate_value(
                item, path=f"{path}.{key}", seen=nested
            )
        return mapping
    if isinstance(value, list | tuple):
        # Wider sequences would silently coerce bytes-like values.
        _reject_cycle(value, path=path, seen=seen)
        nested = (*seen, id(value))
        return [
            _validate_value(item, path=f"{path}[{index}]", seen=nested)
            for index, item in enumerate(value)
        ]
    raise FactError(
        f"step fact {path} is not a JSON value: {type(value).__name__}"
    )


def _reject_cycle(
    container: object,
    *,
    path: str,
    seen: tuple[int, ...],
) -> None:
    if id(container) in seen:
        raise FactError(f"step fact {path} contains a container cycle")


__all__ = [
    "FactError",
    "JsonFactValue",
    "StepFacts",
    "validate_step_facts",
]
