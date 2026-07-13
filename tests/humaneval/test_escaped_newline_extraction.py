from __future__ import annotations

import pytest

from dr_code.humaneval.code_extraction import ExtractionTraceNode
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    CodeParserProfile,
    extract_code_with_profile,
    resolve_parser_profile,
)


def _node_names(nodes: list[ExtractionTraceNode]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        names.add(node.name)
        names.update(_node_names(node.children))
    return names


@pytest.fixture
def v2_profile() -> CodeParserProfile:
    return resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
    )


@pytest.mark.parametrize(
    "source",
    [
        # A: fully escaped and fenced.
        r"Intro\n```python\ndef f():\n    return 1\n```",
        # B: fully escaped and unfenced, including escaped indentation.
        r"Explanation:\ndef f():\n\treturn 1",
        # C: real newlines around an escaped code region.
        "Intro\n" + r"```python\ndef f():\n    return 1\n```",
        # D: the entire response is a JSON-quoted string.
        r'"Intro\n```python\ndef f():\n    return 1\n```"',
    ],
    ids=["escaped-fenced", "escaped-unfenced", "mixed", "json-string"],
)
def test_v2_recovers_escaped_newline_shapes(source: str, v2_profile) -> None:
    result = extract_code_with_profile(source, profile=v2_profile)

    assert result.succeeded
    assert result.extracted_code is not None
    assert "def f():" in result.extracted_code
    assert "unescape_literal_newlines" in _node_names(result.trace.roots)


def test_normal_code_with_string_literal_escape_skips_fallback(v2_profile) -> None:
    # E: a normal code candidate must retain the string-literal escape.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'

    result = extract_code_with_profile(source, profile=v2_profile)

    assert result.succeeded
    assert result.extracted_code == source
    assert "unescape_literal_newlines" not in _node_names(result.trace.roots)


def test_escaped_prose_still_has_no_candidates(v2_profile) -> None:
    # F: applying the fallback does not turn prose into code.
    source = r"Here is a discussion.\nThere is no implementation."

    result = extract_code_with_profile(source, profile=v2_profile)

    assert not result.succeeded
    assert result.extraction_error == "no code candidates extracted"
    assert "unescape_literal_newlines" in _node_names(result.trace.roots)


def test_v1_remains_resolvable_with_historical_behavior() -> None:
    profile = resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version="v1",
    )

    result = extract_code_with_profile(
        r"Intro\ndef f():\n    return 1",
        profile=profile,
    )

    assert not result.succeeded
    assert "unescape_literal_newlines" not in _node_names(result.trace.roots)
