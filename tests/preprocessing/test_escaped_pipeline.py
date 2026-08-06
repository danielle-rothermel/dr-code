from __future__ import annotations

import json

import pytest

from dr_code.core.source.python_analysis import validate_python_source
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
    output = _RUNNER.run(TextArtifact(text=source)).value(OUTPUT_KEY)
    assert isinstance(output, InspectedCodeCandidateSetArtifact), output
    return tuple(item.candidate.source for item in output.candidates)


def test_escaped_pipeline_preserves_string_literal_escape() -> None:
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'
    survivors = _survivors(source)
    assert source in survivors
    for candidate in survivors:
        assert validate_python_source(candidate).compile_ok


@pytest.mark.parametrize(
    "source",
    [
        r"Intro\n```python\ndef f():\n    return 1\n```",
        r"Explanation:\ndef f():\n\treturn 1",
        "Intro\n" + r"```python\ndef f():\n    return 1\n```",
        r'"Intro\n```python\ndef f():\n    return 1\n```"',
    ],
    ids=["escaped-fenced", "escaped-unfenced", "mixed", "json-string"],
)
def test_escaped_pipeline_recovers_escaped_newline_shapes(source: str) -> None:
    survivors = _survivors(source)
    assert any("def f():" in candidate for candidate in survivors)

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
    source = r"Here is a discussion.\nThere is no implementation."
    output = _RUNNER.run(TextArtifact(text=source)).value(OUTPUT_KEY)
    assert is_absent(output)
    assert output.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )


def test_lone_surrogates_remain_a_rejected_input() -> None:
    source = 'def f():\n    return "\ud800"'
    output = _RUNNER.run(TextArtifact(text=source)).value(OUTPUT_KEY)
    assert is_absent(output)
    assert output.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )
