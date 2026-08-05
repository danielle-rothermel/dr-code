"""Validated finite-JSON step facts.

Step facts are descriptive provenance recorded by producers: what a step
chose, why it rejected something, how many candidates survived. They are
persisted verbatim, so the trace boundary restricts them to a closed,
finite JSON shape — string keys, only ``None``/``bool``/``int``/``float``/
``str``/``list``/``dict`` values, finite floats, and no container cycles.

``validate_step_facts`` is the single gate: ``Trace`` construction and the
serialization boundary both run it, so no fact can enter a trace or a
persisted document without having been checked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeAlias

from pydantic import JsonValue

#: The recursive JSON value shape a fact may take. Pydantic's ``JsonValue``
#: already names exactly that shape; ``validate_step_facts`` adds the two
#: constraints it does not express — finite floats and no container cycles.
JsonFactValue: TypeAlias = JsonValue

StepFacts: TypeAlias = Mapping[str, Mapping[str, JsonFactValue]]


class FactError(ValueError):
    """A step fact is outside the finite-JSON shape the trace accepts."""


def validate_step_facts(
    step_facts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, JsonFactValue]]:
    """Deep-copy step facts into plain containers, rejecting non-JSON.

    The returned mapping shares no mutable container with the caller, so a
    later caller mutation cannot change the trace it was recorded in.
    Leaves are narrowed to their plain builtin, so an ``str``/``int``
    subclass such as a ``StrEnum`` member is stored as the ``str``/``int``
    it serializes to rather than as the live enum object.
    """
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
    """Recursively copy one fact value, rejecting non-JSON and cycles.

    ``seen`` carries the identities of the containers currently being
    validated; a repeat identity is a cycle, which has no JSON form.
    """
    if value is None or isinstance(value, bool):
        return value
    # Subclass leaves (``StrEnum``/``IntEnum`` members and the like) are
    # narrowed to their plain builtin, so the stored facts hold only plain
    # containers and never a live domain object.
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
    # Only the two builtin JSON array shapes are accepted. A wider
    # ``Sequence`` test would silently coerce the bytes family — ``bytes``,
    # ``bytearray``, ``memoryview``, ``array.array`` — into lists of ints
    # here while the serialization boundary rejects them.
    if isinstance(value, list | tuple):
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
