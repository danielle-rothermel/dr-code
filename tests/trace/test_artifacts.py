"""Acceptance tests for the Artifact union, Absent, and derived views."""

from __future__ import annotations

import ast

import pytest
from pydantic import TypeAdapter, ValidationError

from dr_code.trace import (
    Absent,
    Artifact,
    ArtifactKind,
    CandidateInspection,
    CandidateLineage,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    IdentifiedCandidate,
    IdentifiedCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    is_absent,
    parsed_module,
)


def _lineage(index: int, candidate_id: str | None = None) -> CandidateLineage:
    return CandidateLineage(
        candidate_id=candidate_id,
        origins=(
            CandidateOrigin(
                path=(
                    ExtractionOperation(
                        kind="test_input", details={"index": index}
                    ),
                )
            ),
        ),
    )


# One representative instance of every artifact kind. The plan gives
# these models in full, so they must construct from their declared
# fields.
ARTIFACT_CASES = [
    TextArtifact(text="hello"),
    CodeArtifact(source="x = 1\n"),
    CodeCandidateSetArtifact(
        candidates=("a = 1", "b = 2"),
        lineage=(_lineage(0), _lineage(1)),
    ),
    IdentifiedCandidateSetArtifact(
        candidates=(
            IdentifiedCandidate(
                source="a = 1",
                lineage=_lineage(0, "candidate-a"),
                inspection=CandidateInspection(
                    parse_ok=True,
                    parse_error=None,
                    compile_ok=True,
                    compile_error=None,
                ),
            ),
        )
    ),
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
    art = CodeCandidateSetArtifact(
        candidates=("first", "second"),
        lineage=(_lineage(0), _lineage(1)),
    )
    # Ordered candidates, conservative first (P-S2).
    assert art.candidates == ("first", "second")
    assert art.kind == ArtifactKind.CODE_CANDIDATE_SET


def test_candidate_set_requires_aligned_lineage() -> None:
    with pytest.raises(ValidationError, match="lineage"):
        CodeCandidateSetArtifact(candidates=("missing",))  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="aligned"):
        CodeCandidateSetArtifact(
            candidates=("first", "second"),
            lineage=(_lineage(0),),
        )


def test_lineage_requires_a_complete_origin_path() -> None:
    with pytest.raises(ValidationError, match="origins"):
        CandidateLineage(candidate_id=None)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="path"):
        CandidateOrigin(path=())


def test_json_artifact_holds_payload() -> None:
    art = JsonArtifact(payload={"nested": [1, 2, 3]})
    assert art.payload == {"nested": [1, 2, 3]}
    assert art.kind == ArtifactKind.JSON


def test_identified_candidate_set_carries_inspection() -> None:
    art = ARTIFACT_CASES[3]
    assert isinstance(art, IdentifiedCandidateSetArtifact)
    assert art.kind == ArtifactKind.IDENTIFIED_CANDIDATE_SET


# --- frozen ----------------------------------------------------------


@pytest.mark.parametrize("art", ARTIFACT_CASES)
def test_artifacts_are_frozen(art: Artifact) -> None:
    # FrozenModel config is frozen=True; mutation must raise.
    with pytest.raises(ValidationError):
        art.kind = ArtifactKind.TEXT  # type: ignore[misc]


def test_absent_is_frozen() -> None:
    absent = Absent(failed_step="s", cause="boom", failure_code="test.failure")
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
            failed_step="s",
            cause="c",
            failure_code="test.failure",
            bogus=1,
        )


def test_absent_requires_failure_code() -> None:
    with pytest.raises(ValidationError, match="failure_code"):
        Absent(failed_step="s", cause="c")  # type: ignore[call-arg]


# --- discriminated parse --------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"kind": "text", "text": "hi"}, TextArtifact),
        ({"kind": "code", "source": "x=1"}, CodeArtifact),
        (
            {
                "kind": "code_candidate_set",
                "candidates": ["a"],
                "lineage": [
                    {
                        "candidate_id": None,
                        "origins": [
                            {
                                "path": [
                                    {
                                        "kind": "test_input",
                                        "details": {"index": 0},
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
            CodeCandidateSetArtifact,
        ),
        (
            {
                "kind": "identified_candidate_set",
                "candidates": [
                    {
                        "source": "a = 1",
                        "lineage": {
                            "candidate_id": "candidate-a",
                            "origins": [
                                {
                                    "path": [
                                        {
                                            "kind": "test_input",
                                            "details": {"index": 0},
                                        }
                                    ]
                                }
                            ],
                        },
                        "inspection": {
                            "parse_ok": True,
                            "parse_error": None,
                            "compile_ok": True,
                            "compile_error": None,
                        },
                    }
                ],
            },
            IdentifiedCandidateSetArtifact,
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
        failure_code="test.syntax_error",
        propagated_through=("score", "aggregate"),
    )
    assert absent.failed_step == "parse"
    assert absent.cause == "syntax error"
    assert absent.propagated_through == ("score", "aggregate")


def test_absent_propagated_through_defaults_empty() -> None:
    absent = Absent(
        failed_step="parse",
        cause="syntax error",
        failure_code="test.syntax_error",
    )
    assert absent.propagated_through == ()


def test_is_absent_true_for_absent() -> None:
    absent = Absent(failed_step="s", cause="c", failure_code="test.failure")
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
    # Derived views are module functions, never fields (S3): the model
    # must not store a parsed AST field.
    assert callable(parsed_module)
    assert "parsed_module" not in CodeArtifact.model_fields
    assert "ast" not in CodeArtifact.model_fields
    assert "module" not in CodeArtifact.model_fields
