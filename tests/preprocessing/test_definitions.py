"""Named preprocessing definitions and the resolver contract.

Contract for ``dr_code.preprocessing.definitions``: frozen
``PreprocessingDefinition`` instances that reproduce the parser profiles
(best-effort v2, best-effort v1, field-marker v2/v1), plus a
``resolve_preprocessing_definition`` that mirrors ``resolve_parser_profile``
— keyword-only, returns the matching frozen definition, raises
``ValueError`` on an unsupported id/version, and matches the old resolver's
permissive pair handling on the off-menu ``(field-marker, v1)`` pair.
"""

from __future__ import annotations

import pytest

from dr_code.humaneval.code_parsing import resolve_parser_profile
from dr_code.preprocessing import (
    PreprocessingDefinition,
    bind_definition,
    preprocessing_definition_hash,
    resolve_preprocessing_definition,
)
from dr_code.preprocessing.definitions import (
    BEST_EFFORT_HUMANEVAL_DEFINITION_ID,
    STRICT_FIELD_MARKER_DEFINITION_ID,
)

BEST_EFFORT_ID = BEST_EFFORT_HUMANEVAL_DEFINITION_ID
FIELD_MARKER_ID = STRICT_FIELD_MARKER_DEFINITION_ID
V1 = "v1"
V2 = "v2"

_SUPPORTED_PAIRS = [
    (BEST_EFFORT_ID, V2),
    (BEST_EFFORT_ID, V1),
    (FIELD_MARKER_ID, V2),
    (FIELD_MARKER_ID, V1),
]


# --- resolution returns the matching definition ----------------------


@pytest.mark.parametrize(
    "definition_id, version",
    _SUPPORTED_PAIRS,
    ids=["best-effort-v2", "best-effort-v1", "field-marker-v2", "field-marker-v1"],
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
    # Mirrors resolve_parser_profile: id and version are keyword-only.
    with pytest.raises(TypeError):
        resolve_preprocessing_definition(BEST_EFFORT_ID, V2)  # type: ignore[misc]


# --- unknown id / version raise ValueError, matching the old resolver -


@pytest.mark.parametrize(
    "definition_id, version",
    [
        ("does-not-exist", V2),
        (BEST_EFFORT_ID, "v99"),
        (FIELD_MARKER_ID, "v99"),
        ("bogus", "bogus"),
    ],
    ids=[
        "unknown-id",
        "best-effort-unknown-version",
        "field-marker-unknown-version",
        "all-unknown",
    ],
)
def test_resolve_rejects_unknown_id_or_version(
    definition_id: str, version: str
) -> None:
    with pytest.raises(ValueError):
        resolve_preprocessing_definition(
            definition_id=definition_id, version=version
        )
    # The old resolver rejects exactly the same coordinates.
    with pytest.raises(ValueError):
        resolve_parser_profile(
            parser_profile_id=definition_id, parser_version=version
        )


def test_resolver_matches_old_on_off_menu_field_marker_v1() -> None:
    # The old resolve_parser_profile accepts (field-marker, v1) — it
    # validates id and version independently, then constructs the profile.
    # Our resolver must match that (a v1 field-marker definition), not
    # substitute v2 (nex-n2-mini's error) or raise.
    old = resolve_parser_profile(
        parser_profile_id=FIELD_MARKER_ID, parser_version=V1
    )
    assert old.profile_id == FIELD_MARKER_ID
    assert old.version == V1

    definition = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version=V1
    )
    assert definition.definition_id == FIELD_MARKER_ID
    assert definition.version == V1
    # Same extraction steps as v2 (the field-marker path never branches on
    # version); only the stamped version differs.
    v2 = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version=V2
    )
    assert definition.steps == v2.steps
    assert definition != v2  # version is part of identity


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


def test_best_effort_v1_and_v2_are_distinct() -> None:
    v1 = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V1
    )
    v2 = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    assert v1 != v2
    assert preprocessing_definition_hash(v1) != preprocessing_definition_hash(v2)


def test_best_effort_v1_v2_differ_only_in_extraction_ladder() -> None:
    # The only axis that varies between v1 and v2 is the extract_candidates
    # strategy tuple: v1 omits the escaped-Python rung.
    v1 = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V1
    )
    v2 = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    assert [s.step for s in v1.steps] == [s.step for s in v2.steps]
    extract_v1 = next(
        s for s in v1.steps if s.step == "extract_candidates"
    )
    extract_v2 = next(
        s for s in v2.steps if s.step == "extract_candidates"
    )
    assert "escaped_python" not in extract_v1.settings["alternatives"]
    assert "escaped_python" in extract_v2.settings["alternatives"]
    # Every other step is identical between the two definitions.
    for a, b in zip(v1.steps, v2.steps, strict=True):
        if a.step == "extract_candidates":
            continue
        assert a == b


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


def test_field_marker_omits_code_repr_filter() -> None:
    # The one asymmetry vs. best-effort: the field-marker path runs
    # include_code_repr_check=False, so no filter_code_repr step.
    fm = resolve_preprocessing_definition(
        definition_id=FIELD_MARKER_ID, version=V2
    )
    be = resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version=V2
    )
    fm_steps = [s.step for s in fm.steps]
    be_steps = [s.step for s in be.steps]
    assert "filter_code_repr" not in fm_steps
    assert "filter_code_repr" in be_steps
    assert "filter_plain_literal" in fm_steps


# --- resolved definitions are bindable and kind-correct --------------


@pytest.mark.parametrize(
    "definition_id, version",
    _SUPPORTED_PAIRS,
    ids=["best-effort-v2", "best-effort-v1", "field-marker-v2", "field-marker-v1"],
)
def test_named_definition_binds(definition_id: str, version: str) -> None:
    definition = resolve_preprocessing_definition(
        definition_id=definition_id, version=version
    )
    # Bind must succeed: the kind chain is internally consistent.
    bound = bind_definition(definition)
    assert bound, "a named definition must have at least one step"
