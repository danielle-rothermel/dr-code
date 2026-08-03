"""Named preprocessing definitions and the resolver contract.

Contract for ``dr_code.preprocessing.definitions``: frozen
``PreprocessingDefinition`` instances for the best-effort and field-marker
extraction paths, plus a ``resolve_preprocessing_definition`` that
is an exact ``(definition_id, version)`` lookup — keyword-only, returns the
matching frozen definition, and raises ``ValueError`` for any pair not in
the table.
"""

from __future__ import annotations

import pytest

from dr_code.preprocessing import (
    PreprocessingDefinition,
    bind_definition,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.definitions import (
    BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
    BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    STRICT_FIELD_MARKER_DEFINITION_ID,
    STRICT_FIELD_MARKER_DEFINITION_VERSION,
)
from dr_code.preprocessing.steps.extract_candidates import (
    ExtractCandidatesSettings,
    ExtractionStrategy,
)

BEST_EFFORT_ID = BEST_EFFORT_HUMANEVAL_DEFINITION_ID
FIELD_MARKER_ID = STRICT_FIELD_MARKER_DEFINITION_ID

_SUPPORTED_PAIRS = [
    (BEST_EFFORT_ID, BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION),
    (FIELD_MARKER_ID, STRICT_FIELD_MARKER_DEFINITION_VERSION),
]


# --- resolution returns the matching definition ----------------------


@pytest.mark.parametrize(
    "definition_id, version",
    _SUPPORTED_PAIRS,
    ids=["best-effort", "field-marker"],
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
        resolve_preprocessing_definition(  # type: ignore[misc]
            BEST_EFFORT_ID, BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION
        )


# --- the resolver raises on every non-registered pair ----------------


@pytest.mark.parametrize(
    "definition_id, version",
    [
        (FIELD_MARKER_ID, "1"),
        (BEST_EFFORT_ID, "1"),
        ("does-not-exist", BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION),
        (BEST_EFFORT_ID, "99"),
        (FIELD_MARKER_ID, "99"),
        ("bogus", "bogus"),
    ],
    ids=[
        "off-menu-field-marker-version",
        "off-menu-best-effort-version",
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


# --- resolved definitions are frozen and distinct --------------------


def test_resolved_definitions_are_frozen() -> None:
    definition = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    with pytest.raises(Exception):
        definition.definition_id = "other"  # type: ignore[misc]


def test_resolution_is_stable() -> None:
    a = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    b = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    assert a == b


def test_named_definitions_are_mutually_distinct() -> None:
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    fm = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID,
        version=STRICT_FIELD_MARKER_DEFINITION_VERSION,
    )
    assert be != fm


# --- filter chains are symmetrical between the two definitions -------


def test_best_effort_and_field_marker_share_filter_chain_order() -> None:
    # Both run plain-literal -> code-repr -> compilable in that order; the
    # field-marker path is no longer missing filter_code_repr.
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    fm = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID,
        version=STRICT_FIELD_MARKER_DEFINITION_VERSION,
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
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    assert "normalize_smart_quotes" in [s.step for s in be.steps]


def test_best_effort_extraction_ladder_has_fourth_rung() -> None:
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID,
        version=BEST_EFFORT_HUMANEVAL_DEFINITION_VERSION,
    )
    extract = next(s for s in be.steps if s.step == "extract_candidates")
    assert isinstance(extract.settings, ExtractCandidatesSettings)
    assert extract.settings.alternatives == (
        ExtractionStrategy.FENCED_BLOCKS,
        ExtractionStrategy.MARKDOWN_WRAPPER,
        ExtractionStrategy.ESCAPED_PYTHON,
        ExtractionStrategy.ESCAPED_MARKDOWN_WRAPPER,
    )


# --- resolved definitions are bindable and kind-correct --------------


@pytest.mark.parametrize(
    "definition_id, version",
    _SUPPORTED_PAIRS,
    ids=["best-effort", "field-marker"],
)
def test_named_definition_binds(definition_id: str, version: str) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=definition_id, version=version
    )
    bound = bind_definition(definition)
    assert bound, "a named definition must have at least one step"
