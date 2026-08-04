"""Boundary tests for trace artifacts and causal absence."""

from __future__ import annotations

import ast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from dr_code.trace import (
    Absent,
    Artifact,
    CodeArtifact,
    CodeCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    is_absent,
    parsed_module,
)

ARTIFACT_PAYLOADS = (
    ({"kind": "text", "text": "hello"}, TextArtifact),
    ({"kind": "code", "source": "x = 1\n"}, CodeArtifact),
    (
        {
            "kind": "code_candidate_set",
            "candidates": ["first", "second"],
        },
        CodeCandidateSetArtifact,
    ),
    ({"kind": "json", "payload": {"task_id": "HumanEval/0"}}, JsonArtifact),
)
ARTIFACTS = (
    TextArtifact(text="hello"),
    CodeArtifact(source="x = 1\n"),
    CodeCandidateSetArtifact(candidates=("first", "second")),
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


def test_absent_preserves_causal_lineage() -> None:
    absent = Absent(
        failed_step="parse",
        cause="syntax error",
        propagated_through=("score", "aggregate"),
    )

    assert absent.model_dump(mode="json") == {
        "kind": "absent",
        "failed_step": "parse",
        "cause": "syntax error",
        "propagated_through": ["score", "aggregate"],
    }
    with pytest.raises(ValidationError):
        absent.cause = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (Absent(failed_step="parse", cause="syntax error"), True),
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
