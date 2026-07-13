from __future__ import annotations

import json

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
    assert "recover_escaped_python" in _node_names(result.trace.roots)


def test_normal_code_with_string_literal_escape_skips_fallback(v2_profile) -> None:
    # E: a normal code candidate must retain the string-literal escape.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'

    result = extract_code_with_profile(source, profile=v2_profile)

    assert result.succeeded
    assert result.extracted_code == source
    assert "recover_escaped_python" not in _node_names(result.trace.roots)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            r'Intro\n```python\ndef join_lines(lines):\n    return "\n".join(lines)\n```',
            'def join_lines(lines):\n    return "\\n".join(lines)',
        ),
        (
            r'Intro\ndef join_lines(lines):\n    return "\n".join(lines)',
            'def join_lines(lines):\n    return "\\n".join(lines)',
        ),
        (
            r'Intro\r```python\rdef join_tabs(parts):\r\treturn "\t".join(parts)\r```',
            'def join_tabs(parts):\n\treturn "\\t".join(parts)',
        ),
        (
            r'Intro\r\n```python\ndef join_cr(parts):\r\n\treturn "\r".join(parts)\n```',
            'def join_cr(parts):\n\treturn "\\r".join(parts)',
        ),
    ],
    ids=[
        "literal-newline-fenced",
        "literal-newline-unfenced",
        "literal-tab",
        "literal-cr-mixed-endings",
    ],
)
def test_escaped_prose_preserves_python_string_literals(
    source: str,
    expected: str,
    v2_profile: CodeParserProfile,
) -> None:
    result = extract_code_with_profile(source, profile=v2_profile)

    assert result.succeeded
    assert result.extracted_code == expected
    assert "recover_escaped_python" in _node_names(result.trace.roots)


def test_json_wrapped_code_preserves_python_string_escapes(v2_profile) -> None:
    expected = (
        'def separators(values):\n'
        '    return "\\n".join(values), "\\t", "\\r"'
    )
    source = json.dumps(f"Intro\n```python\n{expected}\n```")

    result = extract_code_with_profile(source, profile=v2_profile)

    assert result.succeeded
    assert result.extracted_code == expected


def test_escaped_prose_still_has_no_candidates(v2_profile) -> None:
    # F: applying the fallback does not turn prose into code.
    source = r"Here is a discussion.\nThere is no implementation."

    result = extract_code_with_profile(source, profile=v2_profile)

    assert not result.succeeded
    assert result.extraction_error == "no code candidates extracted"
    assert "recover_escaped_python" not in _node_names(result.trace.roots)


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
    assert "recover_escaped_python" not in _node_names(result.trace.roots)
