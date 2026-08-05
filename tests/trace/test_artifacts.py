"""Boundary tests for trace artifacts and causal absence."""

from __future__ import annotations

import ast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from dr_code.trace import (
    Absent,
    Artifact,
    CandidateInspection,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    InspectedCodeCandidate,
    InspectedCodeCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    is_absent,
    parsed_module,
)


def _candidate(source: str, *, location: int = 0) -> CodeCandidate:
    return CodeCandidate(
        source=source,
        origins=(
            CandidateOrigin(
                operation=ExtractionOperation(operation_name="fenced_blocks"),
                input_location=location,
            ),
        ),
    )


#: The exact persisted wire shape of one candidate record, pinned as a
#: literal so a field rename cannot silently change stored identity.
_CANDIDATE_PAYLOAD = {
    "source": "first",
    "origins": [
        {
            "operation": {"operation_name": "fenced_blocks"},
            "input_location": 0,
        }
    ],
}
_SECOND_CANDIDATE_PAYLOAD = {
    "source": "second",
    "origins": [
        {
            "operation": {"operation_name": "fenced_blocks"},
            "input_location": 1,
        }
    ],
}
_INSPECTION_PAYLOAD = {
    "parses": True,
    "parse_error": None,
    "compiles": True,
    "compile_error": None,
    "top_level_function_names": ["f"],
}

ARTIFACT_PAYLOADS = (
    ({"kind": "text", "text": "hello"}, TextArtifact),
    ({"kind": "code", "source": "x = 1\n"}, CodeArtifact),
    (
        {
            "kind": "code_candidate_set",
            "candidates": [_CANDIDATE_PAYLOAD, _SECOND_CANDIDATE_PAYLOAD],
        },
        CodeCandidateSetArtifact,
    ),
    (
        {
            "kind": "inspected_code_candidate_set",
            "candidates": [
                {
                    "candidate": _CANDIDATE_PAYLOAD,
                    "inspection": _INSPECTION_PAYLOAD,
                }
            ],
        },
        InspectedCodeCandidateSetArtifact,
    ),
    ({"kind": "json", "payload": {"task_id": "HumanEval/0"}}, JsonArtifact),
)
ARTIFACTS = (
    TextArtifact(text="hello"),
    CodeArtifact(source="x = 1\n"),
    CodeCandidateSetArtifact(
        candidates=(
            _candidate("first"),
            _candidate("second", location=1),
        )
    ),
    InspectedCodeCandidateSetArtifact(
        candidates=(
            InspectedCodeCandidate(
                candidate=_candidate("first"),
                inspection=CandidateInspection(
                    parses=True,
                    compiles=True,
                    top_level_function_names=("f",),
                ),
            ),
        )
    ),
    JsonArtifact(payload={"task_id": "HumanEval/0"}),
)


@pytest.mark.parametrize(("payload", "expected_type"), ARTIFACT_PAYLOADS)
def test_artifact_union_validates_and_dumps_each_kind(
    payload: dict[str, object],
    expected_type: type[BaseModel],
) -> None:
    adapter = TypeAdapter(Artifact)

    artifact = adapter.validate_python(payload)

    assert isinstance(artifact, expected_type)
    assert adapter.dump_python(artifact, mode="json") == payload


@pytest.mark.parametrize(
    "payload",
    (
        {"kind": "mystery", "text": "hello"},
        {"kind": "text", "text": "hello", "extra": True},
    ),
)
def test_artifact_union_rejects_unknown_contract_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Artifact).validate_python(payload)


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_artifacts_are_immutable(artifact: Artifact) -> None:
    with pytest.raises(ValidationError):
        artifact.kind = "text"  # type: ignore[misc]


def test_candidate_requires_at_least_one_origin() -> None:
    with pytest.raises(ValidationError):
        CodeCandidate(source="x = 1\n", origins=())


def test_candidate_extended_appends_origin_and_keeps_prior_lineage() -> None:
    candidate = _candidate("x = 1\n")
    origin = CandidateOrigin(
        operation=ExtractionOperation(operation_name="dedent_candidates"),
        input_location=3,
    )

    extended = candidate.extended(origin, source="x = 2\n")

    assert extended.source == "x = 2\n"
    assert extended.origins == (*candidate.origins, origin)
    # The original record is untouched: lineage is appended, never replaced.
    assert candidate.origins == candidate.origins[:1]


def test_candidate_inspection_records_structure_without_verdicts() -> None:
    inspection = CandidateInspection(
        parses=False,
        parse_error="SyntaxError: invalid syntax",
        compiles=False,
        compile_error="SyntaxError: invalid syntax",
    )

    assert inspection.model_dump(mode="json") == {
        "parses": False,
        "parse_error": "SyntaxError: invalid syntax",
        "compiles": False,
        "compile_error": "SyntaxError: invalid syntax",
        "top_level_function_names": [],
    }


def test_absent_preserves_causal_lineage() -> None:
    absent = Absent(
        failed_step="parse",
        failure_code="no_candidate_survived_filtering",
        cause="syntax error",
        propagated_through=("score", "aggregate"),
    )

    assert absent.model_dump(mode="json") == {
        "kind": "absent",
        "failed_step": "parse",
        "failure_code": "no_candidate_survived_filtering",
        "cause": "syntax error",
        "propagated_through": ["score", "aggregate"],
    }
    with pytest.raises(ValidationError):
        absent.cause = "other"  # type: ignore[misc]


def test_absent_requires_a_failure_code() -> None:
    with pytest.raises(ValidationError):
        Absent(failed_step="parse", cause="syntax error")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (
            Absent(
                failed_step="parse",
                failure_code="parse_failed",
                cause="syntax error",
            ),
            True,
        ),
        (TextArtifact(text="present"), False),
        ("not a trace value", False),
        (None, False),
    ),
)
def test_is_absent_distinguishes_causal_absence(
    value: object,
    expected: bool,
) -> None:
    assert is_absent(value) is expected


def test_parsed_module_returns_source_ast() -> None:
    module = parsed_module(CodeArtifact(source="x = 1\n"))

    assert isinstance(module, ast.Module)
    assert isinstance(module.body[0], ast.Assign)
