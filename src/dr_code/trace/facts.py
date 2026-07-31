"""Validation helpers for JSON-valued trace facts."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _validate_json_value(
    value: object, *, path: str, active_containers: set[int]
) -> None:
    """Validate one value against the runtime JSON contract."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a container cycle")
        active_containers.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"{path} contains a non-string object key"
                    )
                _validate_json_value(
                    item,
                    path=f"{path}.{key}",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a container cycle")
        active_containers.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return
    raise ValueError(
        f"{path} contains non-JSON value {type(value).__name__!r}"
    )


def validate_step_facts(value: object, *, path: str = "step_facts") -> None:
    """Require the nested object shape used by trace step facts."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    for step_name, facts in value.items():
        if not isinstance(step_name, str):
            raise ValueError(f"{path} contains a non-string step name")
        if not isinstance(facts, Mapping):
            raise ValueError(f"{path}.{step_name} must be an object")
        _validate_json_value(
            facts,
            path=f"{path}.{step_name}",
            active_containers=set(),
        )


__all__ = ["validate_step_facts"]
