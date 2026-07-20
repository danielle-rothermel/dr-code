"""Validation helpers for JSON-valued trace facts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def reject_nonfinite_floats(value: Any, *, path: str = "facts") -> None:
    """Reject values that JSON serialization would silently turn into null."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_nonfinite_floats(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            reject_nonfinite_floats(item, path=f"{path}[{index}]")


__all__ = ["reject_nonfinite_floats"]
