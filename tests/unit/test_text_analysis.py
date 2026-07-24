"""Contract and behavior tests for `dr_code.text_analysis`."""

from __future__ import annotations

import pytest

from dr_code.text_analysis import (
    anchored_code_blocks,
    fence_marker,
    is_code_anchor_line,
    is_code_like_block,
    is_code_like_line,
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
        is_code_like_block,
    ),
    ids=lambda fn: fn.__name__,
)
@pytest.mark.parametrize("source", GARBAGE_INPUTS)
def test_text_analysis_string_functions_are_total(
    analyze, source: str
) -> None:
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


def test_anchored_code_blocks_starts_at_code_anchor() -> None:
    assert anchored_code_blocks("Here is code:\ndef f():\n    return 1") == [
        "def f():\n    return 1"
    ]


def test_anchored_code_blocks_split_functions_separated_by_prose() -> None:
    assert anchored_code_blocks(
        "def first():\n"
        "    return 1\n"
        "\n"
        "This is an explanation, not Python.\n"
        "\n"
        "def second():\n"
        "    return 2",
        segment_prose=True,
    ) == [
        "def first():\n    return 1\n",
        "def second():\n    return 2",
    ]
