"""Artifact-union, absence, and derived-view contracts."""

from __future__ import annotations

import ast

import pytest
from pydantic import TypeAdapter, ValidationError

from dr_code.trace import (
    Absent,
    Artifact,
    ArtifactKind,
    CodeArtifact,
    CodeCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    is_absent,
    parsed_module,
)

# One representative instance of every artifact kind.
ARTIFACT_CASES = [
    TextArtifact(text="hello"),
    CodeArtifact(source="x = 1\n"),
    CodeCandidateSetArtifact(candidates=("a = 1", "b = 2")),
    JsonArtifact(payload={"task_id": "HumanEval/0"}),
]


# --- construction ----------------------------------------------------


def test_text_artifact_constructs() -> None:
    art = TextArtifact(text="hi")
    assert art.text == "hi"
    assert art.kind == ArtifactKind.TEXT


def test_code_artifact_carries_source_only() -> None:
    art = CodeArtifact(source="def f():\n    return 1\n")
    assert art.source == "def f():\n    return 1\n"
    assert art.kind == ArtifactKind.CODE


def test_code_candidate_set_is_ordered_tuple() -> None:
    art = CodeCandidateSetArtifact(candidates=("first", "second"))
    # Candidate order is preserved, with conservative candidates first.
    assert art.candidates == ("first", "second")
    assert art.kind == ArtifactKind.CODE_CANDIDATE_SET


def test_json_artifact_holds_payload() -> None:
    art = JsonArtifact(payload={"nested": [1, 2, 3]})
    assert art.payload == {"nested": [1, 2, 3]}
    assert art.kind == ArtifactKind.JSON


# --- frozen ----------------------------------------------------------


@pytest.mark.parametrize("art", ARTIFACT_CASES)
def test_artifacts_are_frozen(art: Artifact) -> None:
    # FrozenModel config is frozen=True; mutation must raise.
    with pytest.raises(ValidationError):
        art.kind = ArtifactKind.TEXT  # type: ignore[misc]


def test_absent_is_frozen() -> None:
    absent = Absent(failed_step="s", cause="boom")
    with pytest.raises(ValidationError):
        absent.cause = "other"  # type: ignore[misc]


# --- extra fields rejected ------------------------------------------


def test_text_artifact_rejects_extra_fields() -> None:
    # FrozenModel config is extra="forbid".
    with pytest.raises(ValidationError):
        TextArtifact(text="hi", extra="nope")  # type: ignore[call-arg]


def test_absent_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Absent(  # type: ignore[call-arg]
            failed_step="s", cause="c", bogus=1
        )


# --- discriminated parse --------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"kind": "text", "text": "hi"}, TextArtifact),
        ({"kind": "code", "source": "x=1"}, CodeArtifact),
        (
            {"kind": "code_candidate_set", "candidates": ["a"]},
            CodeCandidateSetArtifact,
        ),
        ({"kind": "json", "payload": {"a": 1}}, JsonArtifact),
    ],
)
def test_discriminated_parse_by_kind(
    payload: dict[str, object], expected: type
) -> None:
    adapter = TypeAdapter(Artifact)
    parsed = adapter.validate_python(payload)
    assert isinstance(parsed, expected)


def test_unknown_kind_rejected() -> None:
    # Adding a kind is additive; an unknown kind must not parse.
    adapter = TypeAdapter(Artifact)
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "mystery", "text": "hi"})


# --- Absent lineage --------------------------------------------------


def test_absent_causal_lineage_fields() -> None:
    absent = Absent(
        failed_step="parse",
        cause="syntax error",
        propagated_through=("score", "aggregate"),
    )
    assert absent.failed_step == "parse"
    assert absent.cause == "syntax error"
    assert absent.propagated_through == ("score", "aggregate")


def test_absent_propagated_through_defaults_empty() -> None:
    absent = Absent(failed_step="parse", cause="syntax error")
    assert absent.propagated_through == ()


def test_is_absent_true_for_absent() -> None:
    absent = Absent(failed_step="s", cause="c")
    assert is_absent(absent) is True


@pytest.mark.parametrize("art", ARTIFACT_CASES)
def test_is_absent_false_for_artifacts(art: Artifact) -> None:
    assert is_absent(art) is False


def test_is_absent_false_for_non_trace_value() -> None:
    assert is_absent("not a value") is False
    assert is_absent(None) is False


# --- derived views ---------------------------------------------------


def test_parsed_module_returns_ast_module() -> None:
    art = CodeArtifact(source="x = 1\n")
    module = parsed_module(art)
    assert isinstance(module, ast.Module)


def test_parsed_module_is_a_function_not_a_field() -> None:
    # Derived views are module functions; artifacts do not store parsed ASTs.
    assert callable(parsed_module)
    assert "parsed_module" not in CodeArtifact.model_fields
    assert "ast" not in CodeArtifact.model_fields
    assert "module" not in CodeArtifact.model_fields
