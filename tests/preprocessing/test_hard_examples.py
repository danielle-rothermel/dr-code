"""Central golden coverage for realistic preprocessing hard examples."""

from __future__ import annotations

import ast
from collections import Counter
from functools import lru_cache

import pytest

from dr_code.code_analysis import validate_python_source_with_ast
from dr_code.preprocessing import (
    BoundPreprocessingRunner,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import (
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    TextArtifact,
    is_absent,
)
from hard_examples import (
    AnnotationSource,
    HardExample,
    load_hard_examples,
    partition_for_digest,
)

FIXTURE = load_hard_examples()

DEVELOPMENT_CASES = tuple(
    case for case in FIXTURE.cases if case.partition == "development"
)
HOLDOUT_CASES = tuple(
    case for case in FIXTURE.cases if case.partition == "holdout"
)

INTRINSIC_INVALID_SAMPLE_ID = (
    "007a142c27d875e3697b4cbf449d02febd47a71bd539e46aa6234e3f4bd4f011"
)
CONTRACT_CONFLICT_SAMPLE_IDS = {
    "92ff75341c6dd4dac580b09c886ebd75fc45eecd3ade800b684831e2ba94f34c",
    "9476f3c1b91432ab782dc33aa0e818ae427148050717573e1d64c513c1454a7a",
    "960e6e2c777a759e3464e764e951956b0f7be92dc935b0863aa45e693c67be46",
    "967b8fe0ca2001e94110e5e4f8faafcebf87a9fb439992bf34bf447b56d6ff01",
}


def _case_id(case: HardExample) -> str:
    return case.id


def test_hard_example_fixture_integrity() -> None:
    annotations = [
        source
        for case in FIXTURE.cases
        for source in case.sources
        if isinstance(source, AnnotationSource)
    ]
    annotated_cases = [case for case in FIXTURE.cases if case.annotations]

    assert FIXTURE.schema_version == 1
    assert FIXTURE.partition_algorithm == "sha256-prefix-mod-5-v1"
    assert FIXTURE.annotation_export_checkpoint_sha256 == (
        "c9ebe01e398bfe589fe67b69260553dc37647a8d9a662463cda5c169eb75a441"
    )
    assert FIXTURE.authoritative_corpus_sha256 == (
        "a58acf1b1ed0ad54dc91d12bcca80398f3f3850b559f8051f52af2e4d4f1c4f5"
    )
    assert FIXTURE.annotation_records_sha256 == (
        "0048761890b9e20af9016d15f7b4eacaeb2171bfd895579b89293650063437d5"
    )
    assert len(FIXTURE.cases) == 112
    assert len(annotated_cases) == 91
    assert len(annotations) == 101
    assert Counter(source.verdict for source in annotations) == {
        "should_be_parseable": 35,
        "expected_no_code": 66,
    }
    assert Counter(
        (annotation.verdict, case.expected_outcome)
        for case in annotated_cases
        for annotation in case.annotations
    ) == {
        ("should_be_parseable", "candidates"): 34,
        ("should_be_parseable", "absent"): 1,
        ("expected_no_code", "absent"): 62,
        ("expected_no_code", "candidates"): 4,
    }
    assert Counter(case.expected_outcome for case in annotated_cases) == {
        "candidates": 36,
        "absent": 55,
    }
    assert Counter(case.partition for case in FIXTURE.cases) == {
        "development": 90,
        "holdout": 22,
    }
    assert all(
        case.partition == partition_for_digest(case.decoder_output_sha256)
        for case in FIXTURE.cases
    )
    intact = [
        case for case in annotated_cases if "intact_candidate" in case.categories
    ]
    assert len(intact) == 17
    assert all(
        "drop_after_last_return_salvage"
        in case.forbidden_origin_operation_kinds
        for case in intact
    )
    approved_path_tags = {
        "json_fences",
        "truncated_json",
        "lambda",
        "strange_fences",
        "two_fxns",
    }
    assert all(
        case.required_origin_paths
        for case in annotated_cases
        if case.expected_outcome == "candidates"
        and approved_path_tags.intersection(case.categories)
    )


def test_annotation_adjudications_are_explicit_and_preserve_reviews() -> None:
    cases_by_sample_id = {
        annotation.sample_id: case
        for case in FIXTURE.cases
        for annotation in case.annotations
    }

    invalid = cases_by_sample_id[INTRINSIC_INVALID_SAMPLE_ID]
    assert invalid.annotations[0].verdict == "should_be_parseable"
    assert invalid.adjudication == "intrinsic_invalid"
    assert invalid.expected_outcome == "absent"

    conflicts = {
        sample_id: cases_by_sample_id[sample_id]
        for sample_id in CONTRACT_CONFLICT_SAMPLE_IDS
    }
    assert set(conflicts) == CONTRACT_CONFLICT_SAMPLE_IDS
    assert all(
        case.annotations[0].verdict == "expected_no_code"
        for case in conflicts.values()
    )
    assert all(
        case.adjudication == "contract_conflict"
        and case.expected_outcome == "candidates"
        and "quarantined" in case.categories
        for case in conflicts.values()
    )


def test_exact_fenced_json_spot_check_is_present() -> None:
    case = next(
        case
        for case in FIXTURE.cases
        if case.id == "fenced-json-add-regression"
    )
    assert case.decoder_output_sha256 == (
        "9c0edb516564064842c40c69adb788f88b40826563f0cd0c6b7cd50c2b2a5123"
    )
    assert case.exact_candidates == (
        "def add(a: int, b: int) -> int:\n    return a + b",
    )


def test_forbidden_origin_oracle_rejects_mixed_origins() -> None:
    output = CodeCandidateSetArtifact(
        candidates=("def allowed():\n    return 1", "def mixed():\n    return 2"),
        lineage=(
            CandidateLineage(
                candidate_id="allowed",
                origins=(
                    CandidateOrigin(
                        path=(ExtractionOperation(kind="anchored_python_block"),)
                    ),
                ),
            ),
            CandidateLineage(
                candidate_id="mixed",
                origins=(
                    CandidateOrigin(
                        path=(ExtractionOperation(kind="anchored_python_block"),)
                    ),
                    CandidateOrigin(
                        path=(
                            ExtractionOperation(
                                kind="drop_after_last_return_salvage"
                            ),
                        )
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(AssertionError, match="drop_after_last_return_salvage"):
        _assert_no_forbidden_origin_operations(
            output, {"drop_after_last_return_salvage"}
        )


@pytest.mark.parametrize("case", DEVELOPMENT_CASES, ids=_case_id)
def test_development_hard_examples(case: HardExample) -> None:
    _assert_pipeline_contract(case)


@pytest.mark.parametrize("case", HOLDOUT_CASES, ids=_case_id)
def test_holdout_hard_examples(case: HardExample) -> None:
    _assert_pipeline_contract(case)


def _assert_pipeline_contract(case: HardExample) -> None:
    trace = _runner().run(TextArtifact(text=case.decoder_output))
    output = trace.value("output")

    if case.expected_outcome == "absent":
        assert is_absent(output), _outcome_message(case, output)
        if case.failure_code is not None:
            assert output.failure_code == case.failure_code
        if case.failed_step is not None:
            assert output.failed_step == case.failed_step
        return

    assert isinstance(output, CodeCandidateSetArtifact), _outcome_message(
        case, output
    )
    assert output.candidates
    assert len(output.lineage) == len(output.candidates)
    assert all(item.candidate_id for item in output.lineage)
    assert output.candidates == case.exact_candidates

    actual_function_names: set[str] = set()
    for candidate in output.candidates:
        assert candidate.strip()
        inspected = validate_python_source_with_ast(candidate)
        assert inspected.validation.compile_ok
        assert isinstance(inspected.tree, ast.Module)
        assert any(
            isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            for statement in inspected.tree.body
        )
        actual_function_names.update(
            statement.name
            for statement in inspected.tree.body
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        )
    assert tuple(sorted(actual_function_names)) == (
        case.expected_top_level_function_names
    )

    actual_paths = {
        tuple(
            (
                operation.kind,
                tuple(sorted(operation.details.items())),
            )
            for operation in origin.path
        )
        for lineage in output.lineage
        for origin in lineage.origins
    }
    for required_path in case.required_origin_paths:
        expected_path = tuple(
            (operation.kind, tuple(sorted(operation.details.items())))
            for operation in required_path
        )
        assert expected_path in actual_paths
    forbidden = set(case.forbidden_origin_operation_kinds)
    if forbidden:
        _assert_no_forbidden_origin_operations(output, forbidden)


def _assert_no_forbidden_origin_operations(
    output: CodeCandidateSetArtifact, forbidden: set[str]
) -> None:
    violations = [
        (
            candidate_index,
            origin_index,
            operation_index,
            operation.kind,
        )
        for candidate_index, lineage in enumerate(output.lineage)
        for origin_index, origin in enumerate(lineage.origins)
        for operation_index, operation in enumerate(origin.path)
        if operation.kind in forbidden
    ]
    assert not violations, f"forbidden origin operations found: {violations!r}"


def _outcome_message(case: HardExample, output: object) -> str:
    sample_ids = [annotation.sample_id for annotation in case.annotations]
    return (
        f"{case.id}: expected {case.expected_outcome}; "
        f"sample_ids={sample_ids!r}; actual={output!r}"
    )


@lru_cache(maxsize=1)
def _runner() -> BoundPreprocessingRunner:
    return bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)
