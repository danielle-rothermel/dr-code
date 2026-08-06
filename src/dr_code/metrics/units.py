from __future__ import annotations

from enum import StrEnum, verify, UNIQUE


@verify(UNIQUE)
class MetricFactUnit(StrEnum):
    # Never build payloads by iterating this closed vocabulary.

    COUNT = "count"
    RATIO = "ratio"
    PERCENT = "percent"
    CHARACTERS = "characters"
    BYTES = "bytes"
    LINES = "lines"
    DEPTH = "depth"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    TEXT = "text"


__all__ = ["MetricFactUnit"]
