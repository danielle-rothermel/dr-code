"""Text-normalization preprocessing-step tests."""

from __future__ import annotations

import unicodedata

import pytest

from dr_code.core.source.text_transforms import (
    collapse_blank_runs,
    normalize_line_endings,
    normalize_text,
    strip_trailing_whitespace,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.steps.base import StepFailedError
from dr_code.preprocessing.steps.collapse_blank_runs import CollapseBlankRuns
from dr_code.preprocessing.steps.expand_tabs import ExpandTabs
from dr_code.preprocessing.steps.normalize_line_endings import (
    NormalizeLineEndings,
)
from dr_code.preprocessing.steps.normalize_unicode import NormalizeUnicode
from dr_code.preprocessing.steps.reject_blank_input import RejectBlankInput
from dr_code.preprocessing.steps.strip_trailing_whitespace import (
    StripTrailingWhitespace,
)
from dr_code.preprocessing.steps.trim_outer_blanks import TrimOuterBlanks
from dr_code.trace import TextArtifact


# Garbage inputs reused from the wrapped modules' existing fixtures.
GARBAGE_TEXT = (
    "",
    "def broken(:\n",
    "```\nunterminated fence",
    "plain prose, no code at all",
    "smart ‘quotes’ and “doubles”\r\nCRLF\ttabs  \n\n\n\n",
)

# --- atomic text steps wrap their functions --------------------------


def test_normalize_line_endings_wraps_function() -> None:
    raw = "a\r\nb\rc\n"
    out = NormalizeLineEndings().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=normalize_line_endings(raw))


def test_normalize_unicode_applies_nfkc() -> None:
    raw = "ｄｅｆ"
    out = NormalizeUnicode().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=unicodedata.normalize("NFKC", raw))


def test_expand_tabs_uses_tab_width_setting() -> None:
    raw = "a\tb"
    out = ExpandTabs(ExpandTabs.Settings(tab_width=2)).apply(
        TextArtifact(text=raw)
    )
    assert out.value == TextArtifact(text="a b")


def test_strip_trailing_whitespace_wraps_function() -> None:
    raw = "x = 1  \ny = 2\t\n"
    out = StripTrailingWhitespace().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=strip_trailing_whitespace(raw))


def test_collapse_blank_runs_wraps_function() -> None:
    raw = "a\n\n\n\nb"
    out = CollapseBlankRuns().apply(TextArtifact(text=raw))
    assert out.value == TextArtifact(text=collapse_blank_runs(raw))


def test_trim_outer_blanks_strips_newlines() -> None:
    out = TrimOuterBlanks().apply(TextArtifact(text="\n\nx\n\n"))
    assert out.value == TextArtifact(text="x")


@pytest.mark.parametrize("raw", GARBAGE_TEXT)
def test_atomic_text_sequence_equals_normalize_text(raw: str) -> None:
    """The six atomic steps, in order, reproduce normalize_text."""
    value = TextArtifact(text=raw)
    for step_cls in (
        NormalizeLineEndings,
        NormalizeUnicode,
        ExpandTabs,
        StripTrailingWhitespace,
        CollapseBlankRuns,
        TrimOuterBlanks,
    ):
        value = step_cls().apply(value).value
        assert isinstance(value, TextArtifact)
    assert value.text == normalize_text(raw)


# --- blank-input guard -----------------------------------------------


def test_reject_blank_input_passes_through_non_blank() -> None:
    value = TextArtifact(text="def f(): pass")
    assert RejectBlankInput().apply(value).value == value


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", " \t\n "])
def test_reject_blank_input_fails_on_blank(blank: str) -> None:
    with pytest.raises(StepFailedError) as excinfo:
        RejectBlankInput().apply(TextArtifact(text=blank))
    assert excinfo.value.code is PreprocessingFailureCode.BLANK_INPUT
    assert excinfo.value.evidence == {"input_length": len(blank)}
