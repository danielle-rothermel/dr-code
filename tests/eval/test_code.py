"""Typed code artifacts: candidates, sets, and compile-valid artifacts."""

from __future__ import annotations

import pytest

from dr_code.eval.code import (
    CodeArtifact,
    CodeCandidate,
    CodeCandidateSet,
    CodeCompilationError,
    PythonSource,
)


def test_code_artifact_rejects_uncompilable_source() -> None:
    with pytest.raises(CodeCompilationError) as excinfo:
        CodeArtifact(source="def f(:\n    pass\n")
    assert isinstance(excinfo.value.cause, SyntaxError)


def test_code_artifact_accepts_valid_source() -> None:
    artifact = CodeArtifact(source="def f():\n    return 1\n")
    assert artifact.module().body  # derived view compiles


def test_try_from_candidate_returns_none_on_invalid() -> None:
    good = CodeCandidate(position=0, source="x = 1\n", origin="sf")
    bad = CodeCandidate(position=1, source="x = (\n", origin="sf")
    assert CodeArtifact.try_from_candidate(good) is not None
    assert CodeArtifact.try_from_candidate(bad) is None


def test_candidates_carry_stable_position_and_lineage() -> None:
    candidate_set = CodeCandidateSet.from_sources(
        ("a = 1", "b = 2", "c = 3"), origin="extract"
    )
    assert candidate_set.positions() == (0, 1, 2)
    assert all(c.origin == "extract" for c in candidate_set.candidates)


def test_empty_candidate_set_is_not_a_failure() -> None:
    empty = CodeCandidateSet.from_sources((), origin="extract")
    # An empty set is a valid outcome, distinct from a failure sentinel.
    assert empty.is_empty
    assert empty.candidates == ()


def test_candidate_lineage_defaults_empty_for_legacy_producers() -> None:
    # from_sources stays byte-stable: origin summary, empty rich lineage.
    candidate_set = CodeCandidateSet.from_sources(("a = 1",), origin="ex")
    only = candidate_set.candidates[0]
    assert only.origin == "ex"
    assert only.lineage.candidate_id is None
    assert only.lineage.origins == ()


def test_from_lineage_preserves_multi_origin_lineage_losslessly() -> None:
    from dr_code.trace.artifacts import (
        CandidateLineage,
        CandidateOrigin,
        ExtractionOperation,
    )

    lineage = CandidateLineage(
        candidate_id="c-abc",
        origins=(
            CandidateOrigin(path=(ExtractionOperation(kind="fenced"),)),
            CandidateOrigin(
                path=(ExtractionOperation(kind="last_return_salvage"),)
            ),
        ),
    )
    candidate_set = CodeCandidateSet.from_lineage(
        (("def f():\n    return 1\n", "extract", lineage),)
    )
    only = candidate_set.candidates[0]
    assert only.position == 0
    assert only.origin == "extract"
    # Every origin path survives the crosswalk into the kernel candidate.
    assert only.lineage == lineage
    assert only.lineage.candidate_id == "c-abc"
    assert len(only.lineage.origins) == 2


def test_python_source_is_a_plain_text_role() -> None:
    # PythonSource carries text without a compilation guarantee.
    source = PythonSource(text="def f(:")
    assert source.text == "def f(:"
    candidate = CodeCandidate(position=0, source="def f(:", origin="x")
    assert candidate.to_python_source() == source
