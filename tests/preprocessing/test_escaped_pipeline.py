"""End-to-end escaped-python recovery must not corrupt string literals.

An ``\\n`` inside a Python string literal and an escaped line break in a
JSON-encoded payload look identical; recovery has to decode the structure
without rewriting literal escapes in the code itself.
"""

from __future__ import annotations

import json

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.runner import run_preprocessing
from dr_code.trace import (
    OUTPUT_KEY,
    CodeCandidateSetArtifact,
    TextArtifact,
    is_absent,
)


def _output_source(source: str) -> str:
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
        TextArtifact(text=source),
    )
    output = trace.value(OUTPUT_KEY)
    assert isinstance(output, CodeCandidateSetArtifact)
    assert len(output.candidates) == 1
    return output.candidates[0]


def test_escaped_pipeline_preserves_string_literal_escape() -> None:
    # A normal code candidate must retain the string-literal escape.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'
    output = _output_source(source)
    assert output == source
    assert validate_python_source(output).compile_ok


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
    ],
    ids=["fenced_lf", "unfenced_lf", "cr_tab", "crlf_cr"],
)
def test_escaped_pipeline_preserves_python_string_literals(
    source: str, expected: str
) -> None:
    output = _output_source(source)
    assert output == expected
    assert validate_python_source(output).compile_ok


def test_escaped_pipeline_json_wrapped_code_preserves_string_escapes() -> None:
    expected = (
        'def separators(values):\n    return "\\n".join(values), "\\t", "\\r"'
    )
    source = json.dumps(f"Intro\n```python\n{expected}\n```")
    output = _output_source(source)
    assert output == expected
    assert validate_python_source(output).compile_ok


def test_escaped_pipeline_prose_has_no_candidates() -> None:
    # Applying the fallback does not turn prose into code.
    source = r"Here is a discussion.\nThere is no implementation."
    trace = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize(),
        TextArtifact(text=source),
    )
    assert is_absent(trace.value(OUTPUT_KEY))
