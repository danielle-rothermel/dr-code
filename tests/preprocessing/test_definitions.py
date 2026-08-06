from __future__ import annotations

from typing import Final

import pytest

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID as EXHAUSTIVE_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION as EXHAUSTIVE_VERSION,
    PreprocessingDefinition,
    bind_preprocessing,
    resolve_preprocessing_definition,
)


_EXPECTED_STEP_INSTANCES: Final[tuple[str, ...]] = (
    "normalize_line_endings",
    "normalize_unicode",
    "expand_tabs",
    "strip_trailing_whitespace",
    "collapse_blank_runs",
    "trim_outer_blanks",
    "reject_blank_input",
    "extract_all_representations",
    "strip_fences",
    "dedent",
    "normalize_smart_quotes",
    "split_on_name_guard",
    "add_last_return_salvage",
    "repair_import_lines",
    "infer_missing_imports",
    "dedupe_imports",
    "drop_blank_candidates",
    "dedupe_candidates",
    "inspect_candidates",
    "filter_plain_literal",
    "filter_code_repr",
    "filter_compilable",
    "filter_top_level_functions",
    "materialize_candidate_set",
)


def _definition() -> PreprocessingDefinition:
    return resolve_preprocessing_definition(
        definition_id=EXHAUSTIVE_ID, version=EXHAUSTIVE_VERSION
    )


def test_resolve_returns_the_registered_definition() -> None:
    definition = _definition()
    assert isinstance(definition, PreprocessingDefinition)
    assert definition is EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
    assert (definition.definition_id, definition.version) == (
        EXHAUSTIVE_ID,
        EXHAUSTIVE_VERSION,
    )
    assert (EXHAUSTIVE_ID, EXHAUSTIVE_VERSION) == (
        "exhaustive-function-candidates",
        "0",
    )


def test_resolver_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        resolve_preprocessing_definition(  # type: ignore[misc]
            EXHAUSTIVE_ID, EXHAUSTIVE_VERSION
        )


@pytest.mark.parametrize(
    "definition_id, version",
    [
        (EXHAUSTIVE_ID, "1"),
        (EXHAUSTIVE_ID, "99"),
        ("does-not-exist", EXHAUSTIVE_VERSION),
        ("humaneval-best-effort", "0"),
        ("humaneval-field-marker", "0"),
        ("bogus", "bogus"),
    ],
    ids=[
        "off-menu-version",
        "unknown-version",
        "unknown-id",
        "retired-best-effort-id",
        "retired-field-marker-id",
        "all-unknown",
    ],
)
def test_resolve_rejects_non_registered_pair(
    definition_id: str, version: str
) -> None:
    with pytest.raises(ValueError):
        resolve_preprocessing_definition(
            definition_id=definition_id, version=version
        )


def test_definition_step_instances_are_exactly_pinned() -> None:
    instances = tuple(spec.instance_name for spec in _definition().steps)
    assert instances == _EXPECTED_STEP_INSTANCES


def test_every_source_mutating_step_precedes_inspection() -> None:
    instances = [spec.instance_name for spec in _definition().steps]
    inspection = instances.index("inspect_candidates")
    for mutating in (
        "strip_fences",
        "dedent",
        "normalize_smart_quotes",
        "split_on_name_guard",
        "repair_import_lines",
        "infer_missing_imports",
        "dedupe_imports",
        "add_last_return_salvage",
    ):
        assert instances.index(mutating) < inspection, (
            f"{mutating} rewrites candidate sources and must run before "
            "inspect_candidates"
        )


def test_import_inference_follows_the_last_return_salvage() -> None:
    instances = [spec.instance_name for spec in _definition().steps]

    assert instances.index("add_last_return_salvage") < instances.index(
        "infer_missing_imports"
    )


def test_dedupe_precedes_inspection_and_filters_follow_it() -> None:
    instances = [spec.instance_name for spec in _definition().steps]
    assert (
        instances.index("dedupe_candidates")
        < instances.index("inspect_candidates")
        < instances.index("filter_plain_literal")
    )


def test_materialization_is_the_final_step() -> None:
    assert _definition().steps[-1].instance_name == (
        "materialize_candidate_set"
    )


def test_registered_definition_binds() -> None:
    bound = bind_preprocessing(_definition())
    assert bound.steps
    assert bound.producer.kind == "preprocessing"
