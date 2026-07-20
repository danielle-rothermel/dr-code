"""Public exhaustive preprocessing definition and resolver contract."""

from __future__ import annotations

import pytest

from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    PreprocessingDefinition,
    bind_definition,
    preprocessing_definition_hash,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.definitions import (
    SUPPORTED_DEFINITION_IDS,
    SUPPORTED_DEFINITION_VERSIONS,
)

DEFINITION_ID = HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID
VERSION = "v1"


def _resolve() -> PreprocessingDefinition:
    return resolve_preprocessing_definition(
        definition_id=DEFINITION_ID,
        version=VERSION,
    )


def test_only_function_candidate_definition_is_supported() -> None:
    assert SUPPORTED_DEFINITION_IDS == frozenset({DEFINITION_ID})
    assert SUPPORTED_DEFINITION_VERSIONS == frozenset({VERSION})
    assert _resolve() == HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION


@pytest.mark.parametrize(
    "definition_id,version",
    [
        ("humaneval-best-effort", "v1"),
        ("humaneval-best-effort", "v2"),
        ("humaneval-field-marker", "v2"),
        (DEFINITION_ID, "v2"),
        ("unknown", VERSION),
    ],
)
def test_removed_and_unknown_coordinates_do_not_alias(
    definition_id: str,
    version: str,
) -> None:
    with pytest.raises(ValueError):
        resolve_preprocessing_definition(
            definition_id=definition_id,
            version=version,
        )


def test_resolver_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        resolve_preprocessing_definition(DEFINITION_ID, VERSION)  # type: ignore[misc]


def test_definition_is_stable_hashable_and_bindable() -> None:
    definition = _resolve()
    assert definition.definition_id == DEFINITION_ID
    assert definition.version == VERSION
    assert isinstance(hash(definition), int)
    assert preprocessing_definition_hash(definition) == (
        preprocessing_definition_hash(_resolve())
    )
    assert bind_definition(definition)


def test_definition_orders_exhaustion_cleaning_and_structural_filters() -> None:
    names = [step.instance_name for step in _resolve().steps]
    assert names.index("require_nonblank_text") < names.index(
        "extract_candidates"
    )
    assert names.index("extract_candidates") < names.index("strip_fences")
    assert names.index("dedupe_imports") < names.index(
        "filter_nonblank_candidates"
    )
    assert names.index("filter_nonblank_candidates") < names.index(
        "dedupe_candidates"
    )
    assert names[-5:] == [
        "filter_plain_literal",
        "filter_code_repr",
        "filter_compilable",
        "filter_has_top_level_function",
        "return_all",
    ]
    assert "select_first" not in names
    assert "field_marker_extract" not in names


def test_definition_runs_all_four_discovery_rules() -> None:
    extract = next(
        step
        for step in _resolve().steps
        if step.instance_name == "extract_candidates"
    )
    assert extract.settings["alternatives"] == [
        "fenced_blocks",
        "markdown_wrapper",
        "escaped_python",
        "escaped_markdown_wrapper",
    ]
