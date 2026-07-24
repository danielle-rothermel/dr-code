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
from .hard_examples import (
    AnnotationSource,
    FIXTURE_SHA256,
    FullCorpusRegressionSource,
    HardExample,
    load_hard_examples,
    partition_for_digest,
)

FIXTURE = load_hard_examples()

SEALED_DEVELOPMENT_CASES = tuple(
    case
    for case in FIXTURE.cases
    if case.partition == "development" and case.cohort == "sealed_hard_suite"
)
HOLDOUT_CASES = tuple(
    case
    for case in FIXTURE.cases
    if case.partition == "holdout" and case.cohort == "sealed_hard_suite"
)
POST_HOLDOUT_REGRESSION_CASES = tuple(
    case
    for case in FIXTURE.cases
    if case.cohort == "post_holdout_full_corpus_regression"
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
POST_HOLDOUT_REGRESSION_SAMPLE_IDS = {
    "3c9825e84c0ebe596a68f5766fda5f5487680f502a241fdd8816444cb5cd7f0e",
    "3cfc053f46ec2dec72d6d9505ebb4e8abd9181e46aeabfd5902c9feb32e68fb8",
    "472cd6fe402b9081f562ce78b049389ba7f71fd8ff4c663cbbabaf31f6f45301",
    "4be19b6244fd58d0157cd8da620279ecf942f78783613fb2d27e7071785d2bac",
    "523a2ec1da7dc1cbd76bc3ce656cde7bac8df2019e59dbe3264186952e09a88e",
    "5a4a6240f1810bcdd19bab6e30f6c46f20691d2831666a63020877170745abb3",
    "6caf67db6002a88d36238e220ad0f62ec2227f71df2b577738a5963398b7805f",
    "7ccbdabd030b6df7594aa5e8ce2d06c8546433d509711c33959c131963192bc7",
    "841d8c45e431aa18eb9a2171087cb956a299fea415fe768fed65f0365dff656b",
    "912da4910e0b03c9a673234244ce8f3c9b228aa99d377c2b1da200639a76f5d1",
    "9c20a300c9992eb3c246091242bc366a80eceff60e86cb5d7db03ef85a143dd5",
    "b60847c205a6d35eeb42fccaa0cfd188c2e36fb67df1638ab47dcc4efdf0fce4",
    "bd42e2dd95d73efcfdad0829a2108c764a6d08089c10240d1f3accadba0257cf",
    "c2ae82fcf6f1bbbe4a1e2c5d2124938efbfe46082b1b44a9cb4a9820de2acd6b",
    "cd37b506149034d854f4a65a03ed0b63f839dd6e08de4a7f738c02065c835757",
    "d4f03da500a62e32d5d556c15bf0fdb010609347651b076e684705421262efd7",
    "e05eaa5f5c23217103f486e7e35acc0a0c403da34eafe76193b6189d51106b11",
    "eafeb0a0e810cb29f31ca438258191451a00b34762d8cde1b76d694ddb271b43",
}
ADDITIVE_CONTRACT_CORRECTION_IDS = {
    "annotation-0bb72a8884d3908e",
    "annotation-13e483831955166e",
    "annotation-6ce43080f39f2600",
    "annotation-6f2d0d165ba311f4",
    "annotation-881cb9c30a779e90",
    "annotation-9d15658df4acfb6d",
    "annotation-a83289ac1ebe27bd",
}


def _case_id(case: HardExample) -> str:
    return case.id


def test_hard_example_fixture_integrity() -> None:
    assert FIXTURE_SHA256 == (
        "ed96b9819724fc5b450d37495dd6ca0bbf9fdd3a1304f258631bee34764dc26a"
    )
    annotations = [
        source
        for case in FIXTURE.cases
        for source in case.sources
        if isinstance(source, AnnotationSource)
    ]
    annotated_cases = [case for case in FIXTURE.cases if case.annotations]
    regression_sources = [
        source
        for case in FIXTURE.cases
        for source in case.sources
        if isinstance(source, FullCorpusRegressionSource)
    ]
    sealed_cases = [
        case for case in FIXTURE.cases if case.cohort == "sealed_hard_suite"
    ]

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
    assert FIXTURE.post_holdout_transitions_sha256 == (
        "9a6e4e88f3b1f672616b14cf9490604beeff7413a3508984e2bd10b2ada3b7b6"
    )
    assert FIXTURE.post_holdout_baseline_candidates_sha256 == (
        "64d3effc33089e1fa36aa1db9ce0377e55cf3b324e1e8ab41105c0d99106e560"
    )
    assert len(FIXTURE.cases) == 130
    assert len(sealed_cases) == 112
    assert len(regression_sources) == 18
    assert {source.sample_id for source in regression_sources} == (
        POST_HOLDOUT_REGRESSION_SAMPLE_IDS
    )
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
        "development": 108,
        "holdout": 22,
    }
    assert Counter(case.partition for case in sealed_cases) == {
        "development": 90,
        "holdout": 22,
    }
    assert all(
        case.partition == partition_for_digest(case.decoder_output_sha256)
        for case in sealed_cases
    )
    assert all(
        case.partition == "development"
        and case.adjudication == "production_regression"
        and case.required_origin_operation_kinds
        == ("drop_after_last_return_salvage",)
        and case.required_origin_paths
        and all(
            any(
                operation.kind == "drop_after_last_return_salvage"
                for operation in path
            )
            for path in case.required_origin_paths
        )
        for case in POST_HOLDOUT_REGRESSION_CASES
    )
    salvage_expectations = [
        operation
        for case in FIXTURE.cases
        for path in case.required_origin_paths
        for operation in path
        if operation.kind == "drop_after_last_return_salvage"
    ]
    assert salvage_expectations
    assert all(
        set(operation.details) == {"end_line", "end_column"}
        and isinstance(operation.details["end_line"], int)
        and operation.details["end_line"] >= 1
        and isinstance(operation.details["end_column"], int)
        and operation.details["end_column"] >= 0
        for operation in salvage_expectations
    )
    intact = [
        case
        for case in annotated_cases
        if "intact_candidate" in case.categories
    ]
    corrected_intact = [
        case for case in intact if case.id in ADDITIVE_CONTRACT_CORRECTION_IDS
    ]
    unchanged_intact = [
        case
        for case in intact
        if case.id not in ADDITIVE_CONTRACT_CORRECTION_IDS
    ]
    assert len(intact) == 17
    assert {case.id for case in corrected_intact} == (
        ADDITIVE_CONTRACT_CORRECTION_IDS
    )
    assert all(
        "drop_after_last_return_salvage"
        in case.forbidden_origin_operation_kinds
        for case in unchanged_intact
    )
    assert all(
        not case.forbidden_origin_operation_kinds
        and case.required_origin_paths
        and all(
            path[-1].kind == "drop_after_last_return_salvage"
            for path in case.required_origin_paths
        )
        for case in corrected_intact
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
        candidates=(
            "def allowed():\n    return 1",
            "def mixed():\n    return 2",
        ),
        lineage=(
            CandidateLineage(
                candidate_id="allowed",
                origins=(
                    CandidateOrigin(
                        path=(
                            ExtractionOperation(kind="anchored_python_block"),
                        )
                    ),
                ),
            ),
            CandidateLineage(
                candidate_id="mixed",
                origins=(
                    CandidateOrigin(
                        path=(
                            ExtractionOperation(kind="anchored_python_block"),
                        )
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


def test_additive_contract_corrections_preserve_exact_salvage_paths() -> None:
    """The written additive contract and corpus regression override suppression."""
    corrected = [
        case
        for case in FIXTURE.cases
        if case.id in ADDITIVE_CONTRACT_CORRECTION_IDS
    ]

    for case in corrected:
        output = (
            _runner()
            .run(TextArtifact(text=case.decoder_output))
            .value("output")
        )
        assert isinstance(output, CodeCandidateSetArtifact)
        assert output.candidates == case.exact_candidates

        salvage_paths: set[
            tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
        ] = set()
        salvage_started = False
        for index, (candidate, lineage) in enumerate(
            zip(output.candidates, output.lineage, strict=True)
        ):
            candidate_has_salvage = any(
                operation.kind == "drop_after_last_return_salvage"
                for origin in lineage.origins
                for operation in origin.path
            )
            if not candidate_has_salvage:
                assert not salvage_started
                continue
            salvage_started = True
            assert all(
                origin.path[-1].kind == "drop_after_last_return_salvage"
                for origin in lineage.origins
            )
            assert any(
                prior != candidate and prior.startswith(candidate)
                for prior in output.candidates[:index]
            )
            salvage_paths.update(
                tuple(
                    (
                        operation.kind,
                        tuple(sorted(operation.details.items())),
                    )
                    for operation in origin.path
                )
                for origin in lineage.origins
            )

        expected_paths = {
            tuple(
                (
                    operation.kind,
                    tuple(sorted(operation.details.items())),
                )
                for operation in path
            )
            for path in case.required_origin_paths
        }
        assert salvage_started
        assert salvage_paths == expected_paths


@pytest.mark.parametrize("case", SEALED_DEVELOPMENT_CASES, ids=_case_id)
def test_development_hard_examples(case: HardExample) -> None:
    _assert_pipeline_contract(case)


@pytest.mark.parametrize("case", HOLDOUT_CASES, ids=_case_id)
def test_holdout_hard_examples(case: HardExample) -> None:
    _assert_pipeline_contract(case)


@pytest.mark.parametrize("case", POST_HOLDOUT_REGRESSION_CASES, ids=_case_id)
def test_post_holdout_full_corpus_regressions(case: HardExample) -> None:
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
    required = set(case.required_origin_operation_kinds)
    if required:
        assert all(
            required.issubset(operation.kind for operation in origin.path)
            for lineage in output.lineage
            for origin in lineage.origins
        )
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
