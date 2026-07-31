from __future__ import annotations

import json

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    CodeParserProfile,
    resolve_parser_profile,
)
from dr_code.preprocessing import (
    resolve_preprocessing_definition,
    run_preprocessing,
)
from dr_code.trace import CodeArtifact, TextArtifact, is_absent


@pytest.fixture
def v2_profile() -> CodeParserProfile:
    return resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=PARSER_PROFILE_VERSION,
    )


def _extract(source: str, profile: CodeParserProfile) -> str | None:
    definition = resolve_preprocessing_definition(
        definition_id=profile.profile_id,
        version=profile.version,
    )
    output = run_preprocessing(
        definition.materialize(),
        TextArtifact(text=source),
    ).value("output")
    if is_absent(output):
        return None
    assert isinstance(output, CodeArtifact)
    return output.source


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
    extracted = _extract(source, v2_profile)

    assert extracted is not None
    assert "def f():" in extracted
    # Round-trip: recovery is only correct if the recovered code compiles.
    assert validate_python_source(extracted).compile_ok


def test_normal_code_with_string_literal_escape_skips_fallback(
    v2_profile,
) -> None:
    # E: a normal code candidate must retain the string-literal escape.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'

    extracted = _extract(source, v2_profile)

    assert extracted == source
    assert validate_python_source(extracted).compile_ok


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
    extracted = _extract(source, v2_profile)

    assert extracted == expected
    # Round-trip: preserved string literals must still compile.
    assert validate_python_source(extracted).compile_ok


def test_json_wrapped_code_preserves_python_string_escapes(v2_profile) -> None:
    expected = (
        'def separators(values):\n    return "\\n".join(values), "\\t", "\\r"'
    )
    source = json.dumps(f"Intro\n```python\n{expected}\n```")

    extracted = _extract(source, v2_profile)

    assert extracted == expected
    # Round-trip: preserved string literals must still compile.
    assert validate_python_source(extracted).compile_ok


def test_escaped_prose_still_has_no_candidates(v2_profile) -> None:
    # F: applying the fallback does not turn prose into code.
    source = r"Here is a discussion.\nThere is no implementation."

    extracted = _extract(source, v2_profile)

    assert extracted is None


def test_v1_parser_profile_is_not_resolvable() -> None:
    with pytest.raises(ValueError, match="unsupported parser profile version"):
        resolve_parser_profile(
            parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
            parser_version="v1",
        )
