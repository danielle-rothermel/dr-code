"""Contract and behavior tests for `dr_code.text_transforms`."""

from __future__ import annotations

import pytest

from dr_code.text_transforms import (
    collapse_blank_runs,
    fence_marker,
    is_code_anchor_line,
    is_code_like_line,
    normalize_line_endings,
    normalize_smart_quotes,
    normalize_text,
    strip_code_fences,
    strip_markdown_wrappers,
    strip_trailing_whitespace,
    wrap_code_fence,
)

GARBAGE_INPUTS = (
    "",
    "def broken(:\n",
    "```\nunterminated fence",
    "plain prose, no code at all",
    "smart ‘quotes’ and “doubles”\r\nCRLF\ttabs  \n\n\n\n",
)

TOTAL_TRANSFORMS = (
    collapse_blank_runs,
    normalize_line_endings,
    normalize_smart_quotes,
    normalize_text,
    strip_code_fences,
    strip_markdown_wrappers,
    strip_trailing_whitespace,
    wrap_code_fence,
)


@pytest.mark.parametrize("transform", TOTAL_TRANSFORMS, ids=lambda fn: fn.__name__)
@pytest.mark.parametrize("source", GARBAGE_INPUTS)
def test_text_transforms_are_total(transform, source: str) -> None:
    assert isinstance(transform(source), str)


def test_normalize_line_endings_converts_crlf_and_cr() -> None:
    assert normalize_line_endings("a\r\nb\rc\n") == "a\nb\nc\n"


def test_strip_trailing_whitespace_per_line() -> None:
    assert strip_trailing_whitespace("x = 1  \ny = 2\t\n") == "x = 1\ny = 2\n"


def test_collapse_blank_runs_to_one_blank_line() -> None:
    assert collapse_blank_runs("a\n\n\n\nb") == "a\n\nb"


def test_normalize_text_folds_crlf_tabs_unicode_and_blanks() -> None:
    raw = "ｄｅｆ f():\r\n\treturn 1  \r\n\r\n\r\n\r\nx = 2\r\n"
    out = normalize_text(raw)
    assert out == "def f():\n    return 1\n\nx = 2"


def test_wrap_then_strip_code_fences_round_trips() -> None:
    source = "def f():\n    return 1\n"
    assert strip_code_fences(wrap_code_fence(source)) == source


def test_wrap_code_fence_untagged_when_lang_empty() -> None:
    assert wrap_code_fence("x = 1", "") == "```\nx = 1\n```\n"


def test_strip_code_fences_leaves_unfenced_text_alone() -> None:
    assert strip_code_fences("x = 1\ny = 2") == "x = 1\ny = 2"


def test_fence_marker_detects_fence_lines_only() -> None:
    assert fence_marker("```python") == "```"
    assert fence_marker("~~~") == "~~~"
    assert fence_marker("x = 1") is None


def test_strip_markdown_wrappers_removes_one_marker_per_line() -> None:
    text = "> quoted\n1. numbered\n- bullet\nplain"
    assert strip_markdown_wrappers(text) == "quoted\nnumbered\nbullet\nplain"


def test_normalize_smart_quotes_restores_ascii() -> None:
    assert normalize_smart_quotes("s = ‘a’ + “b”") == "s = 'a' + \"b\""


def test_is_code_anchor_line() -> None:
    assert is_code_anchor_line("def f():")
    assert is_code_anchor_line("from os import path")
    assert not is_code_anchor_line("Here is the solution:")


def test_is_code_like_line_treats_blank_as_code_like() -> None:
    assert is_code_like_line("")
    assert is_code_like_line("    return 1")
    assert not is_code_like_line("This sentence is prose.")
