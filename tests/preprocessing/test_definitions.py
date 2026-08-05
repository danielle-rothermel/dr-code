"""The registered definition and the resolver contract.

Contract for ``dr_code.preprocessing.definitions``: one frozen
``PreprocessingDefinition`` describing the exhaustive function-candidate
pipeline, plus a ``resolve_preprocessing_definition`` that is an exact
``(definition_id, version)`` lookup — keyword-only, returns the matching
frozen definition, and raises ``ValueError`` for any pair not in the table.
"""

from __future__ import annotations

from typing import Final

import pytest

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID as EXHAUSTIVE_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION as EXHAUSTIVE_VERSION,
    PreprocessingDefinition,
    bind_preprocessing,
    resolve_preprocessing_definition,
)

#: The definition's exact ordered step-instance names. This pins the whole
#: pipeline shape: which representations are read, that cleaning precedes
#: inspection, that salvage is additive, and that materialization is last.
#: A change here is a change to what the definition *is* and must be a
#: deliberate, versioned decision.
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
    "repair_import_lines",
    "infer_missing_imports",
    "dedupe_imports",
    "add_last_return_salvage",
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


# --- resolution returns the matching definition ----------------------


def test_resolve_returns_the_registered_definition() -> None:
    definition = _definition()
    assert isinstance(definition, PreprocessingDefinition)
    assert definition.definition_id == EXHAUSTIVE_ID
    assert definition.version == EXHAUSTIVE_VERSION


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


# --- the resolved definition is frozen and stable --------------------


def test_resolved_definition_is_frozen() -> None:
    with pytest.raises(Exception):
        _definition().definition_id = "other"  # type: ignore[misc]


def test_resolution_is_stable() -> None:
    assert _definition() == _definition()


# --- the definition owns its own coordinate --------------------------


def test_definition_id_is_preprocessing_owned() -> None:
    # The id names what the pipeline does, not a dataset that consumes it:
    # extraction behavior is independent of who scores against it.
    assert EXHAUSTIVE_ID == "exhaustive-function-candidates"
    assert EXHAUSTIVE_VERSION == "0"


# --- the step chain is exactly this, in this order -------------------


def test_definition_step_instances_are_exactly_pinned() -> None:
    instances = tuple(spec.instance_name for spec in _definition().steps)
    assert instances == _EXPECTED_STEP_INSTANCES


def test_every_source_mutating_step_precedes_inspection() -> None:
    # The inspection must describe the exact source it accompanies, so no
    # step that rewrites a source may run after inspection. Import
    # inference is the load-bearing case: it prepends import lines.
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


# --- the resolved definition binds ------------------------------------


def test_registered_definition_binds() -> None:
    bound = bind_preprocessing(_definition())
    assert bound.steps
    assert bound.producer.kind == "preprocessing"
