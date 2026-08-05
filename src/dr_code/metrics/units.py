"""The closed unit vocabulary metric facts are measured in.

Every operator declares the unit of each fact it emits at its own
definition, so the unit travels with the fact into the persisted record and
consumers never have to infer dimensionality from a fact name.

Never build a payload by iterating this enum: the set of members is a closed
vocabulary, not an ordered list, and its iteration order is not part of any
persisted format. Reference members individually by name.
"""

from __future__ import annotations

from enum import StrEnum, verify, UNIQUE


@verify(UNIQUE)
class MetricFactUnit(StrEnum):
    """Every unit a metric fact may be measured in."""

    #: A cardinality: how many of something were observed.
    COUNT = "count"
    #: A dimensionless quotient of two like quantities.
    RATIO = "ratio"
    #: A quotient rendered on a zero-to-one-hundred scale.
    PERCENT = "percent"
    #: A length measured in Unicode characters.
    CHARACTERS = "characters"
    #: A length measured in bytes.
    BYTES = "bytes"
    #: A length measured in source lines.
    LINES = "lines"
    #: A count of nesting levels.
    DEPTH = "depth"
    #: A truth value.
    BOOLEAN = "boolean"
    #: A name identifying something in the measured artifact.
    IDENTIFIER = "identifier"
    #: Free-form text emitted as an observation, not as a measurement.
    TEXT = "text"


__all__ = ["MetricFactUnit"]
