"""Named preprocessing definitions and the resolver contract.

Contract for ``dr_code.preprocessing.definitions``: frozen
``PreprocessingDefinition`` instances for the best-effort and field-marker
extraction paths (v2 only), plus a ``resolve_preprocessing_definition`` that
is an exact ``(definition_id, version)`` lookup — keyword-only, returns the
matching frozen definition, and raises ``ValueError`` for any pair not in
the table.
"""

from __future__ import annotations

import pytest

from dr_code.preprocessing import (
    PreprocessingDefinition,
    bind_definition,
    preprocessing_definition_hash,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.definitions import (
    BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
    STRICT_FIELD_MARKER_DEFINITION_ID,
    SUPPORTED_DEFINITION_VERSIONS,
)

BEST_EFFORT_ID = BEST_EFFORT_HUMANEVAL_DEFINITION_ID
FIELD_MARKER_ID = STRICT_FIELD_MARKER_DEFINITION_ID
V1 = "v1"
V2 = "v2"

_SUPPORTED_PAIRS = [
    (BEST_EFFORT_ID, V2),
    (FIELD_MARKER_ID, V2),
]


# --- resolution returns the matching definition ----------------------


@pytest.mark.parametrize(
    "definition_id, version",
    _SUPPORTED_PAIRS,
    ids=["best-effort-v2", "field-marker-v2"],
)
def test_resolve_returns_named_definition(
    definition_id: str, version: str
) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=definition_id, version=version
    )
    assert isinstance(definition, PreprocessingDefinition)
    assert definition.definition_id == definition_id
    assert definition.version == version


def test_resolver_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        resolve_preprocessing_definition(BEST_EFFORT_ID, V2)  # type: ignore[misc]


def test_supported_versions_is_v2_only() -> None:
    assert SUPPORTED_DEFINITION_VERSIONS == frozenset({V2})


# --- the resolver raises on every non-registered pair ----------------


@pytest.mark.parametrize(
    "definition_id, version",
    [
        (FIELD_MARKER_ID, V1),
        (BEST_EFFORT_ID, V1),
        ("does-not-exist", V2),
        (BEST_EFFORT_ID, "v99"),
        (FIELD_MARKER_ID, "v99"),
        ("bogus", "bogus"),
    ],
    ids=[
        "off-menu-field-marker-v1",
        "off-menu-best-effort-v1",
        "unknown-id",
        "best-effort-unknown-version",
        "field-marker-unknown-version",
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


# --- resolved definitions are frozen / hashable / distinct -----------


def test_resolved_definitions_are_frozen() -> None:
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    with pytest.raises(Exception):
        definition.definition_id = "other"  # type: ignore[misc]


def test_resolved_definitions_are_hashable() -> None:
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    assert isinstance(hash(definition), int)
    assert isinstance(preprocessing_definition_hash(definition), str)


def test_resolution_is_stable() -> None:
    a = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    b = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    assert a == b
    assert preprocessing_definition_hash(a) == preprocessing_definition_hash(b)


def test_named_definitions_are_mutually_distinct() -> None:
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    fm = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version=V2
    )
    assert be != fm


# --- filter chains are symmetrical between the two definitions -------


def test_best_effort_and_field_marker_share_filter_chain_order() -> None:
    # Both run plain-literal -> code-repr -> compilable in that order; the
    # field-marker path is no longer missing filter_code_repr.
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    fm = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version=V2
    )
    filter_chain = [
        "filter_plain_literal",
        "filter_code_repr",
        "filter_compilable",
    ]
    be_filters = [s.step for s in be.steps if s.step in filter_chain]
    fm_filters = [s.step for s in fm.steps if s.step in filter_chain]
    assert be_filters == filter_chain
    assert fm_filters == filter_chain


# --- best-effort carries the smart-quote step and the fourth rung ----


def test_best_effort_includes_smart_quote_step() -> None:
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    assert "normalize_smart_quotes" in [s.step for s in be.steps]


def test_best_effort_extraction_ladder_has_fourth_rung() -> None:
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    extract = next(s for s in be.steps if s.step == "extract_candidates")
    assert extract.settings["alternatives"] == [
        "fenced_blocks",
        "markdown_wrapper",
        "escaped_python",
        "escaped_markdown_wrapper",
    ]


# --- resolved definitions are bindable and kind-correct --------------


@pytest.mark.parametrize(
    "definition_id, version",
    _SUPPORTED_PAIRS,
    ids=["best-effort-v2", "field-marker-v2"],
)
def test_named_definition_binds(definition_id: str, version: str) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=definition_id, version=version
    )
    bound = bind_definition(definition)
    assert bound, "a named definition must have at least one step"
