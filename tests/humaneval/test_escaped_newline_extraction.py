from __future__ import annotations

from dr_code.core.source.python_analysis import validate_python_source
from dr_code.humaneval.acceptance import extract_humaneval_code
from dr_code.preprocessing import PreprocessingFailureCode


def test_normal_code_with_string_literal_escape_is_accepted() -> None:
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'

    result = extract_humaneval_code(source)

    assert result.succeeded
    assert result.accepted_code == source
    assert validate_python_source(result.accepted_code).compile_ok


def test_escaped_prose_preserves_python_string_literals() -> None:
    source = (
        r"Intro\n```python\ndef join_lines(lines):"
        r'\n    return "\n".join(lines)\n```'
    )
    expected = 'def join_lines(lines):\n    return "\\n".join(lines)'

    result = extract_humaneval_code(source)

    assert result.succeeded
    assert result.accepted_code == expected

    assert validate_python_source(result.accepted_code).compile_ok


def test_escaped_prose_still_has_no_candidates() -> None:
    source = r"Here is a discussion.\nThere is no implementation."

    result = extract_humaneval_code(source)

    assert not result.succeeded
    assert result.accepted_code is None
    assert result.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )
