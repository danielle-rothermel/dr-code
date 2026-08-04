"""String-aware smart-quote recovery over code candidates.

Unicode "smart" quotes reach LLM code output both as string *delimiters*
(which must become ASCII to compile) and as literal *contents* inside an
ASCII-quoted string (which must be preserved — they are program data).
A blanket translation corrupts the latter, so this step converts smart
quotes only where the scanner is outside an ASCII-quoted string literal,
tracking single/double/triple-quote state and backslash escapes in the
spirit of ``text_transforms._decode_python_structure``.
"""

from __future__ import annotations

from typing import ClassVar

from dr_code.text_transforms import SMART_QUOTES
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep

#: Smart quote character -> its ASCII counterpart.
_SMART_TO_ASCII: dict[str, str] = {
    smart: ascii_quote
    for ascii_quote, pair in SMART_QUOTES.items()
    for smart in pair
}
_ASCII_QUOTES: frozenset[str] = frozenset({"'", '"'})


def _normalize_outside_ascii_strings(source: str) -> str:
    """Convert smart quotes except inside ASCII-quoted string literals.

    An ASCII quote inside a ``#`` comment (e.g. ``# don't``) must not open
    string state — comment text is not a string delimiter. Smart quotes in
    comments are converted; comments carry no program semantics.
    """
    parts: list[str] = []
    quote: str | None = None
    triple_quoted = False
    in_comment = False
    index = 0
    length = len(source)
    while index < length:
        character = source[index]

        if quote is not None:
            if source.startswith("\\", index):
                parts.append(source[index : index + 2])
                index += 2
                continue
            closing = quote * (3 if triple_quoted else 1)
            if source.startswith(closing, index):
                parts.append(closing)
                index += len(closing)
                quote = None
                triple_quoted = False
                continue
            parts.append(character)
            index += 1
            continue

        if character == "\n":
            in_comment = False
        elif not in_comment and character == "#":
            in_comment = True
        elif not in_comment and character in _ASCII_QUOTES:
            quote = character
            triple_quoted = source.startswith(character * 3, index)
            opening = character * (3 if triple_quoted else 1)
            parts.append(opening)
            index += len(opening)
            continue

        parts.append(_SMART_TO_ASCII.get(character, character))
        index += 1

    return "".join(parts)


class NormalizeSmartQuotes(CandidateMapStep):
    """Convert smart quotes to ASCII outside ASCII-quoted string literals."""

    NAME: ClassVar[StepName] = StepName.NORMALIZE_SMART_QUOTES
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        return _normalize_outside_ascii_strings(source)


__all__ = ["NormalizeSmartQuotes"]
