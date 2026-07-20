"""End-to-end contract cases for the exhaustive named definition."""

from __future__ import annotations

import json

import pytest

from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import CodeCandidateSetArtifact, TextArtifact, is_absent

RUNNER = bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)


def _run(raw: str):
    return RUNNER.run(TextArtifact(text=raw))


@pytest.mark.parametrize(
    "raw,expected_code",
    [
        ("def f():\n    return 1\n", "def f():\n    return 1"),
        (
            "Here is code:\n```python\ndef f():\n    return 1\n```\n",
            "def f():\n    return 1",
        ),
        (
            json.dumps({"code": "def from_json():\n    return 2\n"}),
            "def from_json():\n    return 2",
        ),
        (
            "[[ ## code ## ]]\ndef from_marker():\n    return 3\n"
            "[[ ## completed ## ]]",
            "def from_marker():\n    return 3",
        ),
        (
            "def greet():\n    return “hello”",
            'def greet():\n    return "hello"',
        ),
        (
            "async def fetch():\n    return 1\n",
            "async def fetch():\n    return 1",
        ),
    ],
)
def test_named_pipeline_recovers_function_candidate(
    raw: str,
    expected_code: str,
) -> None:
    output = _run(raw).value("output")
    assert isinstance(output, CodeCandidateSetArtifact)
    assert expected_code in output.candidates
    assert output.candidates
    assert all(item.candidate_id for item in output.lineage)


@pytest.mark.parametrize(
    "raw,failure_code",
    [
        ("", "decoder_output_blank"),
        (" \n\t", "decoder_output_blank"),
        ("This is only prose.", "no_code_candidates"),
        ("```python\n{1: 2, 3: 4}\n```", "plain_literal_only"),
        ('code = "def f(): pass"\n', "code_repr_only"),
        ("def broken(:\n", "no_compilable_candidate"),
        ("x = 1\n", "no_top_level_function_candidate"),
    ],
)
def test_named_pipeline_emits_specific_terminal_failure(
    raw: str,
    failure_code: str,
) -> None:
    output = _run(raw).value("output")
    assert is_absent(output)
    assert output.failure_code == failure_code


def test_function_name_is_not_a_preprocessing_policy() -> None:
    output = _run("def any_name_is_valid():\n    return 1\n").value("output")
    assert isinstance(output, CodeCandidateSetArtifact)
    assert output.candidates


def test_multiple_functions_are_returned_without_first_candidate_selection() -> None:
    raw = (
        "```python\ndef first():\n    return 1\n```\n"
        "```python\ndef second():\n    return 2\n```\n"
    )
    output = _run(raw).value("output")
    assert isinstance(output, CodeCandidateSetArtifact)
    assert any("def first" in source for source in output.candidates)
    assert any("def second" in source for source in output.candidates)
