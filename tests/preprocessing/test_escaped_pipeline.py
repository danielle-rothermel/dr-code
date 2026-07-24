"""End-to-end escaped-python recovery must not corrupt string literals.

An ``\\n`` inside a Python string literal and an escaped line break in a
JSON-encoded payload look identical; recovery has to decode the structure
without rewriting literal escapes in the code itself.

This is the one home for escaped-newline behavioral coverage. The pipeline
below is a deliberately minimal escaped-recovery ladder — normalization,
extraction, fence stripping, salvage, import handling, and compilability —
so a failure names the recovery behavior rather than a policy filter. The
registered ``humaneval-function-candidates`` definition runs the same steps;
cross-API agreement is asserted in ``test_parity``.
"""

from __future__ import annotations

import json

import pytest

from dr_code.code_analysis import validate_python_source
from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.runner import run_external_preprocessing
from dr_code.trace import (
    OUTPUT_KEY,
    CodeCandidateSetArtifact,
    TextArtifact,
    is_absent,
)


def _escaped_pipeline_definition() -> PreprocessingDefinition:
    """Full normalization, extraction, and compilability definition.

    Import inference happens inside ``identify_candidates``, which already
    holds each candidate's parsed tree; ``expand_last_return_salvage``
    carries the trailing-return salvage behavior.
    """

    def _spec(name: str, step: StepName) -> StepSpec:
        return StepSpec(instance_name=name, step=step)

    return PreprocessingDefinition(
        definition_id="escaped",
        version="1",
        steps=(
            _spec("le", StepName.NORMALIZE_LINE_ENDINGS),
            _spec("unicode", StepName.NORMALIZE_UNICODE),
            _spec("tabs", StepName.EXPAND_TABS),
            _spec("strip", StepName.STRIP_TRAILING_WHITESPACE),
            _spec("blank", StepName.COLLAPSE_BLANK_RUNS),
            _spec("trim", StepName.TRIM_OUTER_BLANKS),
            _spec("extract", StepName.EXTRACT_CANDIDATES),
            _spec("fences", StepName.STRIP_FENCES),
            _spec("split", StepName.SPLIT_ON_NAME_GUARD),
            _spec("salvage", StepName.EXPAND_LAST_RETURN_SALVAGE),
            _spec("repair", StepName.REPAIR_IMPORT_LINES),
            _spec("dedupe", StepName.DEDUPE_IMPORTS),
            _spec("identify", StepName.IDENTIFY_CANDIDATES),
            _spec("filter", StepName.FILTER_COMPILABLE),
            _spec("materialize", StepName.MATERIALIZE_CANDIDATES),
            _spec("all", StepName.RETURN_ALL),
        ),
    )


def _candidates(source: str) -> tuple[str, ...]:
    trace = run_external_preprocessing(
        _escaped_pipeline_definition(), TextArtifact(text=source)
    )
    output = trace.value(OUTPUT_KEY)
    assert isinstance(output, CodeCandidateSetArtifact)
    return output.candidates


def _only_candidate(source: str) -> str:
    """The single candidate an unambiguous escaped payload recovers to."""
    candidates = _candidates(source)
    assert len(candidates) == 1, candidates
    return candidates[0]


def test_escaped_pipeline_preserves_string_literal_escape() -> None:
    # A normal code candidate must retain the string-literal escape.
    source = 'def join_lines(lines):\n    return "\\n".join(lines)'
    recovered = _only_candidate(source)
    assert recovered == source
    assert validate_python_source(recovered).compile_ok


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
    candidates = _candidates(source)
    assert candidates
    # Round-trip: recovery is only correct if every recovered candidate
    # carries the function and compiles.
    for candidate in candidates:
        assert "def f():" in candidate
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
    recovered = _only_candidate(source)
    assert recovered == expected
    assert validate_python_source(recovered).compile_ok


def test_escaped_pipeline_json_wrapped_code_preserves_string_escapes() -> None:
    expected = (
        'def separators(values):\n    return "\\n".join(values), "\\t", "\\r"'
    )
    source = json.dumps(f"Intro\n```python\n{expected}\n```")
    recovered = _only_candidate(source)
    assert recovered == expected
    assert validate_python_source(recovered).compile_ok


def test_escaped_pipeline_prose_has_no_candidates() -> None:
    # Applying the fallback does not turn prose into code.
    source = r"Here is a discussion.\nThere is no implementation."
    trace = run_external_preprocessing(
        _escaped_pipeline_definition(), TextArtifact(text=source)
    )
    assert is_absent(trace.value(OUTPUT_KEY))
