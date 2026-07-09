"""Contract and behavior tests for `dr_code.text_analysis`."""

from __future__ import annotations

import pytest

from dr_code.text_analysis import (
    anchored_code_blocks,
    candidate_blocks,
    code_like_blocks,
    fence_marker,
    is_code_anchor_line,
    is_code_like_block,
    is_code_like_line,
    split_by_fences,
)

GARBAGE_INPUTS = (
    "",
    "def broken(:\n",
    "```\nunterminated fence",
    "plain prose, no code at all",
)


@pytest.mark.parametrize(
    "analyze",
    (
        anchored_code_blocks,
        candidate_blocks,
        is_code_like_block,
        split_by_fences,
    ),
    ids=lambda fn: fn.__name__,
)
@pytest.mark.parametrize("source", GARBAGE_INPUTS)
def test_text_analysis_string_functions_are_total(analyze, source: str) -> None:
    analyze(source)


def test_text_analysis_line_functions_are_total() -> None:
    for source in GARBAGE_INPUTS:
        assert isinstance(fence_marker(source), str | None)
        assert isinstance(is_code_anchor_line(source), bool)
        assert isinstance(is_code_like_line(source), bool)


def test_fence_marker_detects_fence_lines_only() -> None:
    assert fence_marker("```python") == "```"
    assert fence_marker("~~~") == "~~~"
    assert fence_marker("x = 1") is None


def test_is_code_anchor_line() -> None:
    assert is_code_anchor_line("def f():")
    assert is_code_anchor_line("from os import path")
    assert not is_code_anchor_line("Here is the solution:")


def test_is_code_like_line_treats_blank_as_code_like() -> None:
    assert is_code_like_line("")
    assert is_code_like_line("    return 1")
    assert not is_code_like_line("This sentence is prose.")


def test_split_by_fences_prefers_matching_closer_and_keeps_unmatched_fenced() -> None:
    text = "before\n```python\nx = 1\n~~~\ny = 2\n```\nafter"

    unfenced, fenced = split_by_fences(text)

    assert unfenced == ["before", "after"]
    assert fenced == ["x = 1\n~~~\ny = 2"]


def test_candidate_blocks_prefers_fenced_blocks() -> None:
    text = "before\n```python\nx = 1\n```\nafter"

    assert candidate_blocks(text) == ["x = 1"]


def test_anchored_code_blocks_starts_at_code_anchor() -> None:
    assert anchored_code_blocks("Here is code:\ndef f():\n    return 1") == [
        "def f():\n    return 1"
    ]


def test_code_like_blocks_flattens_anchored_segments() -> None:
    assert code_like_blocks(["prose\ndef f():\n    return 1"]) == [
        "def f():\n    return 1"
    ]
