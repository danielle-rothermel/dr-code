"""Normalize decoder text before it reaches preprocessing boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NormalizedDecoderOutput:
    """A lossless JSON-safe decoder representation and its validity facts."""

    text: str
    contains_nul: bool
    contains_surrogate: bool

    @property
    def is_valid(self) -> bool:
        """Whether the original decoder text can safely reach Python tools."""
        return not self.contains_nul and not self.contains_surrogate

    @property
    def facts(self) -> dict[str, bool]:
        """Return stable validation facts without recording the input text."""
        return {
            "contains_nul": self.contains_nul,
            "contains_surrogate": self.contains_surrogate,
        }


def normalize_decoder_output(text: str) -> NormalizedDecoderOutput:
    """Make invalid decoder code points visible and safe for result storage.

    NUL and lone surrogate code points cannot safely enter the tokenizer,
    UTF-8 candidate identity, or a persisted JSON result. Their escapes retain
    the exact offending code point for diagnostics while the validity facts
    ensure they are never interpreted as Python source.
    """
    normalized: list[str] = []
    contains_nul = False
    contains_surrogate = False
    for character in text:
        if character == "\x00":
            contains_nul = True
            normalized.append("\\x00")
        elif "\ud800" <= character <= "\udfff":
            contains_surrogate = True
            normalized.append(f"\\u{ord(character):04x}")
        else:
            normalized.append(character)
    return NormalizedDecoderOutput(
        text="".join(normalized),
        contains_nul=contains_nul,
        contains_surrogate=contains_surrogate,
    )


__all__ = ["NormalizedDecoderOutput", "normalize_decoder_output"]
