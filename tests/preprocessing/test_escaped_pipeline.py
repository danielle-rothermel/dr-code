"""End-to-end escaped-python recovery must not corrupt string literals.

An ``\\n`` inside a Python string literal and an escaped line break in a
JSON-encoded payload look identical; recovery has to decode the structure
without rewriting literal escapes in the code itself.

These assertions run against the registered definition, and read the
materialized candidate set rather than a single chosen value: recovery is
correct when the recovered source is *present* among the survivors, not
when it happens to be the one a consumer would accept.
"""

from __future__ import annotations

import json

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    PreprocessingFailureCode,
    bind_preprocessing,
)
from dr_code.trace import (
    OUTPUT_KEY,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    is_absent,
)

_RUNNER = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)


def _survivors(source: str) -> tuple[str, ...]:
    """Every candidate source the registered definition materialized."""
    output = _RUNNER.run(TextArtifact(text=source)).value(OUTPUT_KEY)
    assert isinstance(output, InspectedCodeCandidateSetArtifact), output
    return tuple(item.candidate.source for item in output.candidates)


def test_escaped_pipeline_preserves_string_literal_escape() -> None:
    # A normal code candidate must retain the string-literal escape.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'
    survivors = _survivors(source)
    assert source in survivors
    for candidate in survivors:
        assert validate_python_source(candidate).compile_ok


@pytest.mark.parametrize(
    "source",
    [
        # Fully escaped and fenced.
        r"Intro\n```python\ndef f():\n    return 1\n```",
        # Fully escaped and unfenced, including escaped indentation.
        r"Explanation:\ndef f():\n\treturn 1",
        # Real newlines around an escaped code region.
        "Intro\n" + r"```python\ndef f():\n    return 1\n```",
        # The entire response is a JSON-quoted string.
        r'"Intro\n```python\ndef f():\n    return 1\n```"',
    ],
    ids=["escaped-fenced", "escaped-unfenced", "mixed", "json-string"],
)
def test_escaped_pipeline_recovers_escaped_newline_shapes(source: str) -> None:
    survivors = _survivors(source)
    assert any("def f():" in candidate for candidate in survivors)
    # Round-trip: recovery is only correct if the recovered code compiles.
    for candidate in survivors:
        assert validate_python_source(candidate).compile_ok


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            r"Intro\n```python\ndef join_lines(lines):"
            r'\n    return "\n".join(lines)\n```',
            'def join_lines(lines):\n    return "\\n".join(lines)',
        ),
        (
            r'Intro\ndef join_lines(lines):\n    return "\n".join(lines)',
            'def join_lines(lines):\n    return "\\n".join(lines)',
        ),
        (
            r"Intro\r```python\rdef join_tabs(parts):"
            r'\r\treturn "\t".join(parts)\r```',
            'def join_tabs(parts):\n\treturn "\\t".join(parts)',
        ),
        (
            r"Intro\r\n```python\r\ndef join_cr(parts):"
            r'\r\n\treturn "\r".join(parts)\n```',
            'def join_cr(parts):\n\treturn "\\r".join(parts)',
        ),
        # An escaped CRLF intro followed by an LF-escaped fence opener: the
        # two escaped line-ending forms mix within one payload.
        (
            r"Intro\r\n```python\ndef join_cr(parts):"
            r'\r\n\treturn "\r".join(parts)\n```',
            'def join_cr(parts):\n\treturn "\\r".join(parts)',
        ),
    ],
    ids=[
        "fenced_lf",
        "unfenced_lf",
        "cr_tab",
        "crlf_cr",
        "mixed_endings_fence_lf",
    ],
)
def test_escaped_pipeline_preserves_python_string_literals(
    source: str, expected: str
) -> None:
    survivors = _survivors(source)
    assert expected in survivors
    assert validate_python_source(expected).compile_ok


def test_escaped_pipeline_json_wrapped_code_preserves_string_escapes() -> None:
    expected = (
        'def separators(values):\n    return "\\n".join(values), "\\t", "\\r"'
    )
    source = json.dumps(f"Intro\n```python\n{expected}\n```")
    assert expected in _survivors(source)


def test_escaped_pipeline_prose_has_no_candidates() -> None:
    # Recovering escapes does not turn prose into code.
    source = r"Here is a discussion.\nThere is no implementation."
    output = _RUNNER.run(TextArtifact(text=source)).value(OUTPUT_KEY)
    assert is_absent(output)
    assert output.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )


def test_lone_surrogates_remain_a_rejected_input() -> None:
    # No step silently rewrites candidate content, so a lone surrogate is
    # carried through to compilation and rejected there rather than being
    # normalized away into something that compiles.
    source = 'def f():\n    return "\ud800"'
    output = _RUNNER.run(TextArtifact(text=source)).value(OUTPUT_KEY)
    assert is_absent(output)
    assert output.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )
