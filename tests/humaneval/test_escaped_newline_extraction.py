from __future__ import annotations

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION,
    CodeParserProfile,
    extract_code_with_profile,
    resolve_parser_profile,
)


@pytest.fixture
def current_profile() -> CodeParserProfile:
    return resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION,
    )


def test_normal_code_with_string_literal_escape_skips_fallback(
    current_profile,
) -> None:
    # A normal code candidate must retain the string-literal escape: the
    # escaped-newline fallback rung stays unused when the candidate already
    # compiles.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'

    result = extract_code_with_profile(source, profile=current_profile)

    assert result.succeeded
    assert result.extracted_code == source
    assert validate_python_source(result.extracted_code).compile_ok


def test_escaped_prose_preserves_python_string_literals(
    current_profile: CodeParserProfile,
) -> None:
    # One representative shape on this API; the behavioral matrix of escaped
    # line-ending shapes lives in tests/preprocessing/test_escaped_pipeline.py
    # and the two APIs are not compared on these inputs elsewhere.
    source = (
        r"Intro\n```python\ndef join_lines(lines):"
        r'\n    return "\n".join(lines)\n```'
    )
    expected = 'def join_lines(lines):\n    return "\\n".join(lines)'

    result = extract_code_with_profile(source, profile=current_profile)

    assert result.succeeded
    assert result.extracted_code == expected
    # Round-trip: preserved string literals must still compile.
    assert validate_python_source(result.extracted_code).compile_ok


def test_escaped_prose_still_has_no_candidates(current_profile) -> None:
    # Applying the fallback does not turn prose into code, and the parser
    # reports the miss as an extraction error rather than empty code.
    source = r"Here is a discussion.\nThere is no implementation."

    result = extract_code_with_profile(source, profile=current_profile)

    assert not result.succeeded
    assert result.extraction_error == "no code candidates extracted"
