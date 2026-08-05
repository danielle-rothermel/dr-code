from __future__ import annotations

import ast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from dr_code.core.source.python_analysis import validate_python_source_with_ast
from dr_code.trace import (
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


# Literal payloads pin persisted artifact keys and candidate identity.
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


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_json_artifact_rejects_nested_non_finite_floats(
    value: float,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        JsonArtifact(payload={"outer": [{"value": value}]})

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "value_error"
    assert error["loc"] == ("payload",)
    assert str(error["ctx"]["error"]) == (
        "JSON artifact payload must contain only finite floats"
    )


@pytest.mark.parametrize("artifact", ARTIFACTS)
def test_artifacts_are_immutable(artifact: Artifact) -> None:
    with pytest.raises(ValidationError):
        artifact.kind = "text"  # type: ignore[misc]


def test_candidate_requires_at_least_one_origin() -> None:
    with pytest.raises(ValidationError):
        CodeCandidate(source="x = 1\n", origins=())


def test_candidate_extended_appends_origin_and_keeps_prior_lineage() -> None:
    candidate = _candidate("x = 1\n")
    origins_before = candidate.origins
    origin = CandidateOrigin(
        operation=ExtractionOperation(operation_name="dedent_candidates"),
        input_location=3,
    )

    extended = candidate.extended(origin, source="x = 2\n")

    assert extended.source == "x = 2\n"
    assert extended.origins == (*origins_before, origin)

    assert candidate.origins == origins_before
    assert candidate.source == "x = 1\n"


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


def test_candidate_origin_rejects_negative_input_location() -> None:
    with pytest.raises(ValidationError):
        CandidateOrigin(
            operation=ExtractionOperation(operation_name="strip_fences"),
            input_location=-1,
        )


@pytest.mark.parametrize(
    ("fields", "reason"),
    (
        (
            {
                "parses": False,
                "parse_error": "SyntaxError: invalid syntax",
                "compiles": True,
            },
            "compiles without parsing",
        ),
        (
            {"parses": False, "compiles": False, "compile_error": "boom"},
            "missing parse_error when parses is False",
        ),
        (
            {
                "parses": True,
                "parse_error": "SyntaxError: invalid syntax",
                "compiles": True,
            },
            "parse_error present when parses is True",
        ),
        (
            {"parses": True, "compiles": False},
            "missing compile_error when compiles is False",
        ),
        (
            {"parses": True, "compiles": True, "compile_error": "boom"},
            "compile_error present when compiles is True",
        ),
    ),
)
def test_candidate_inspection_rejects_impossible_structural_facts(
    fields: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(ValidationError):
        CandidateInspection(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "source",
    (
        "def f():\n    return 1\n",
        "",
        "def f(:\n",
        'def f():\n    return "\ud800"',
        "x = 1\x00",
    ),
)
def test_source_validation_only_yields_valid_inspections(source: str) -> None:
    validation = validate_python_source_with_ast(source).validation

    inspection = CandidateInspection(
        parses=validation.parse_ok,
        parse_error=validation.parse_error,
        compiles=validation.compile_ok,
        compile_error=validation.compile_error,
    )

    assert inspection.parses == (inspection.parse_error is None)
    assert inspection.compiles == (inspection.compile_error is None)
    assert not (inspection.compiles and not inspection.parses)


def test_parsed_module_returns_source_ast() -> None:
    module = parsed_module(CodeArtifact(source="x = 1\n"))

    assert isinstance(module, ast.Module)
    assert isinstance(module.body[0], ast.Assign)
