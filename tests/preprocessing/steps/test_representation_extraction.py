from __future__ import annotations

import json

import pytest

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.steps.base import StepFailedError
from dr_code.preprocessing.steps.extract_all_representations import (
    ExtractAllRepresentations,
    Representation,
)
from dr_code.trace import CodeCandidateSetArtifact, TextArtifact


def _sources(value: CodeCandidateSetArtifact) -> tuple[str, ...]:
    return tuple(candidate.source for candidate in value.candidates)


def _extract(text: str):
    return ExtractAllRepresentations().apply(TextArtifact(text=text))


def _origin_operations(value: CodeCandidateSetArtifact) -> list[str]:
    return [
        candidate.origins[0].operation.operation_name
        for candidate in value.candidates
    ]


def test_extraction_reads_fenced_and_raw_representations_together() -> None:
    out = _extract("Intro\n```python\ndef f():\n    return 1\n```")
    operations = _origin_operations(out.value)
    assert Representation.RAW_RESPONSE.value in operations
    assert Representation.TEXT_SEGMENTS.value in operations
    assert "def f():\n    return 1" in _sources(out.value)


def test_extraction_reads_fenced_and_unfenced_code_additively() -> None:
    out = _extract(
        "def outside():\n    return 1\n\n"
        "```python\ndef inside():\n    return 2\n```"
    )

    segment_sources = [
        candidate.source
        for candidate in out.value.candidates
        if candidate.origins[0].operation.operation_name
        == Representation.TEXT_SEGMENTS.value
    ]
    assert "def inside():\n    return 2" in segment_sources
    assert "def outside():\n    return 1\n" in segment_sources


def test_extraction_reads_unfenced_segments() -> None:
    out = _extract("Explanation first.\ndef f():\n    return 1")
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.TEXT_SEGMENTS.value in _origin_operations(out.value)


def test_extraction_reads_markdown_wrapped_segments() -> None:
    out = _extract("> def f():\n>     return 1")
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.MARKDOWN_SEGMENTS.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_whole_response_json_string() -> None:
    out = _extract(json.dumps("def f():\n    return 1"))
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_STRING_RESPONSE.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_top_level_json_code_field() -> None:
    out = _extract(json.dumps({"code": "def f():\n    return 1"}))
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value in _origin_operations(
        out.value
    )


@pytest.mark.parametrize("tag", ["json", ""])
def test_extraction_reads_a_json_code_field_inside_a_fence(tag: str) -> None:
    envelope = json.dumps({"code": "def f():\n    return 1"})
    out = _extract(f"Here it is:\n\n```{tag}\n{envelope}\n```\n\nDone.")
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_fenced_json_code_field_once() -> None:
    envelope = json.dumps({"code": "def f():\n    return 1"})
    out = _extract(f"```json\n{envelope}\n```")
    sources = [
        candidate.source
        for candidate in out.value.candidates
        if candidate.origins[0].operation.operation_name
        == Representation.JSON_CODE_FIELD.value
    ]
    assert len(sources) == len(set(sources))


def test_extraction_ignores_a_malformed_fenced_json_envelope() -> None:
    out = _extract('```json\n{"code": "def f():\\n    return 1"\n```')
    assert Representation.JSON_CODE_FIELD.value not in _origin_operations(
        out.value
    )


def test_extraction_ignores_a_fenced_envelope_outside_the_code_field() -> None:
    envelope = json.dumps({"code": "def reference():\n    return 999"})
    out = _extract(
        f"[[ ## prompt ## ]]\nFor example:\n\n```json\n{envelope}\n```\n\n"
        "[[ ## code ## ]]\ndef f():\n    return 1\n"
    )
    assert "def reference():\n    return 999" not in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value not in _origin_operations(
        out.value
    )
    assert out.value.candidates[0].source == "def f():\n    return 1"


def test_extraction_reads_a_fenced_envelope_inside_the_code_field() -> None:
    envelope = json.dumps({"code": "def f():\n    return 1"})
    out = _extract(
        f"[[ ## prompt ## ]]\nWhat?\n[[ ## code ## ]]\n```json\n{envelope}\n```"
    )
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.JSON_CODE_FIELD.value in _origin_operations(
        out.value
    )


def test_extraction_reads_a_field_marker_value() -> None:
    out = _extract(
        "[[ ## prompt ## ]]\nWhat?\n[[ ## code ## ]]\ndef f():\n    return 1\n"
    )
    assert "def f():\n    return 1" in _sources(out.value)
    assert Representation.FIELD_MARKER.value in _origin_operations(out.value)


def test_extraction_reads_escaped_python() -> None:
    out = _extract(r"Explanation:\ndef f():\n\treturn 1")
    assert Representation.ESCAPED_PYTHON.value in _origin_operations(out.value)


def test_extraction_reads_escaped_markdown() -> None:
    out = _extract(json.dumps("- def add(a, b):\n-     return a + b"))
    assert "def add(a, b):\n    return a + b" in _sources(out.value)
    assert Representation.ESCAPED_MARKDOWN.value in _origin_operations(
        out.value
    )


def test_extraction_contributes_in_declared_representation_order() -> None:
    out = _extract("Intro\n```python\ndef f():\n    return 1\n```")
    order = [Representation(name) for name in _origin_operations(out.value)]
    declared = list(Representation)
    positions = [declared.index(item) for item in order]
    assert positions == sorted(positions)


def test_extraction_records_per_representation_counts_as_facts() -> None:
    out = _extract("```python\ndef f():\n    return 1\n```")
    for representation in Representation:
        assert representation.value in out.facts
    assert out.facts["candidate_count"] == len(out.value.candidates)


def test_extraction_with_no_readable_representation_fails() -> None:
    with pytest.raises(StepFailedError) as excinfo:
        _extract("   ")
    assert (
        excinfo.value.code is PreprocessingFailureCode.NO_CANDIDATES_EXTRACTED
    )

    assert set(excinfo.value.evidence) == {
        representation.value for representation in Representation
    }
    assert set(excinfo.value.evidence.values()) == {0}


def test_extraction_of_prose_yields_the_raw_response_only() -> None:
    out = _extract("This is an explanation with no code whatsoever.")
    assert _origin_operations(out.value) == [Representation.RAW_RESPONSE.value]
