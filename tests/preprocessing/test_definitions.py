"""Public exhaustive preprocessing definition and resolver contract."""

from __future__ import annotations

import pytest

from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID,
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    PreprocessingDefinition,
    bind_definition,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.definitions import (
    DEFINITION_VERSION,
    SUPPORTED_DEFINITION_IDS,
    SUPPORTED_DEFINITION_VERSIONS,
)
from dr_code.preprocessing.registry import REGISTRY

DEFINITION_ID = HUMANEVAL_FUNCTION_CANDIDATES_DEFINITION_ID
VERSION = DEFINITION_VERSION


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
        (DEFINITION_ID, "1"),
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


def test_definition_is_stable_and_bindable() -> None:
    definition = _resolve()
    assert definition.definition_id == DEFINITION_ID
    assert definition.version == VERSION
    # Resolution is by coordinate, so the resolver returns an equal
    # canonical definition every time.
    assert definition == _resolve()
    assert bind_definition(definition)


def test_definition_orders_exhaustion_cleaning_and_structural_filters() -> (
    None
):
    names = [step.instance_name for step in _resolve().steps]
    assert names.index("require_nonblank_text") < names.index(
        "extract_candidates"
    )
    assert names.index("extract_candidates") < names.index("strip_fences")
    assert names.index("dedupe_imports") < names.index(
        "filter_nonblank_candidates"
    )
    assert names.index("filter_nonblank_candidates") < names.index(
        "identify_candidates"
    )
    assert names[-6:] == [
        "filter_plain_literal",
        "filter_code_repr",
        "filter_compilable",
        "filter_has_top_level_function",
        "materialize_candidates",
        "return_all",
    ]
    assert len(names) == 23
    assert {spec.step for spec in _resolve().steps} == set(REGISTRY)


def test_extraction_step_carries_no_settings() -> None:
    # Extraction is exhaustive: it has no alternatives knob to configure,
    # so its persisted coordinate carries an empty settings projection.
    extract = next(
        step
        for step in _resolve().steps
        if step.instance_name == "extract_candidates"
    )
    assert extract.settings.model_dump() == {}


def test_every_step_in_the_definition_is_registered() -> None:
    # Binding resolves each spec through the registry, so an unregistered
    # step name could never reach a run.
    for spec in _resolve().steps:
        assert spec.step.value in REGISTRY
