"""Corruption round-trips over the frozen recipe set.

Applying each ``dr_code.synthetic`` recipe to clean HumanEval-shaped code
and running the registered definition partitions the recipes empirically:

* ``RECOVERABLE`` — formatting / noise / import / wrapper pathologies the
  pipeline undoes. The recovered code is *equivalent* to the original
  (``dr_code.core.source.python_analysis.equivalent``). Because the definition
  materializes every survivor rather than choosing one, the assertion is
  that an equivalent candidate is *among* the survivors — recovery is a
  question of whether the pipeline found the code, not of which survivor a
  particular consumer would accept.

* ``NON_RECOVERABLE`` — the corruption makes a semantic or structural
  change no extraction pipeline can undo, so no survivor is ever
  equivalent to ground truth:

  - ``inline_backticks`` destroys all code-like structure — no candidate
    survives, so the run yields ``Absent`` (the pipeline gives up rather
    than fabricate code).
  - ``dead_code`` / ``renamed_locals`` / ``string_form_swap`` rewrite the
    AST (extra statements, renamed bindings, swapped string forms), so a
    survivor compiles but is not semantically equivalent to the original.
  - ``truncated_midfn`` / ``truncated_and_unfenced`` break the source at an
    RNG-chosen point, so the outcome is seed-dependent and never reliably
    equivalent.

  For every recipe in this set we still assert something meaningful: that
  no survivor is equivalent to ground truth, which is what makes the
  exemption from the equivalence assertion warranted rather than a gap.
  No exempted recipe sits in an assertion-free bucket.
"""

from __future__ import annotations

import random

import pytest

from dr_code.core.source.python_analysis import equivalent
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_preprocessing,
)
from dr_code.synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
from dr_code.trace import (
    OUTPUT_KEY,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    is_absent,
)

SEED = 0

CLEAN = (
    "import numpy as np\n"
    "\n"
    "def make_array(values):\n"
    "    total = 0\n"
    "    for v in values:\n"
    "        total += v\n"
    "    return np.array([total, total])\n"
)

#: Recipes whose corruption the registered definition undoes, recovering a
#: candidate equivalent to the original.
RECOVERABLE_RECIPES = [
    "clean",
    "fenced_tagged",
    "fenced_untagged",
    "fenced_with_prose",
    "chat_indented",
    "smart_quoted",
    "crlf_tabs",
    "missing_np_import",
    "mangled_import_paren",
    "two_solutions",
    "markdown_blockquote",
    "unicode_fullwidth",
    "kitchen_sink",
    "trailing_whitespace",
    "blank_lines_noise",
    "duplicated_imports",
    "comments_noise",
    "quote_style_swap",
    "extra_type_annotations",
]

#: Recipes that make an unrecoverable semantic/structural change (see the
#: module docstring). Exempted from equivalence; asserted non-equivalent.
NON_RECOVERABLE_RECIPES = [
    "inline_backticks",
    "dead_code",
    "renamed_locals",
    "string_form_swap",
    "truncated_midfn",
    "truncated_and_unfenced",
]

_RUNNER = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)


def _corrupt(recipe_name: str) -> str:
    recipe = RECIPES_BY_NAME[recipe_name]
    return apply_recipe(recipe, CLEAN, random.Random(SEED)).corrupted_source


def _survivors(raw: str) -> tuple[str, ...] | None:
    """Materialized candidate sources, or ``None`` when the run is Absent."""
    output = _RUNNER.run(TextArtifact(text=raw)).value(OUTPUT_KEY)
    if is_absent(output):
        return None
    assert isinstance(output, InspectedCodeCandidateSetArtifact)
    return tuple(item.candidate.source for item in output.candidates)


# --- recoverable recipes round-trip to equivalent code ---------------


@pytest.mark.parametrize("recipe_name", RECOVERABLE_RECIPES)
def test_recoverable_corruption_yields_an_equivalent_candidate(
    recipe_name: str,
) -> None:
    survivors = _survivors(_corrupt(recipe_name))
    assert survivors is not None, (
        f"{recipe_name}: expected surviving candidates, got Absent"
    )
    assert any(equivalent(CLEAN, source) for source in survivors), (
        f"{recipe_name}: no surviving candidate is equivalent to the "
        f"original; survivors={survivors!r}"
    )


# --- non-recoverable recipes recover nothing equivalent --------------


@pytest.mark.parametrize("recipe_name", NON_RECOVERABLE_RECIPES)
def test_non_recoverable_corruption_recovers_nothing_equivalent(
    recipe_name: str,
) -> None:
    # No extraction pipeline can recover equivalence here. Asserting that
    # *no* survivor is equivalent is what confirms the exemption is
    # warranted rather than masking a recoverable case.
    survivors = _survivors(_corrupt(recipe_name))
    if survivors is None:
        return
    assert not any(equivalent(CLEAN, source) for source in survivors), (
        f"{recipe_name}: unexpectedly recovered an equivalent candidate; "
        f"it should be reclassified as recoverable"
    )


def test_inline_backticks_yields_absent() -> None:
    # The strongest give-up case: inline backticks destroy all code-like
    # structure, so no candidate survives and the run is Absent.
    assert _survivors(_corrupt("inline_backticks")) is None


# --- every candidate the pipeline returns actually compiles ----------


@pytest.mark.parametrize(
    "recipe_name", RECOVERABLE_RECIPES + NON_RECOVERABLE_RECIPES
)
def test_every_survivor_compiles(recipe_name: str) -> None:
    # The compilability filter reads stored inspections; this asserts the
    # stored inspection actually described the source it accompanied.
    from dr_code.core.source.python_analysis import validate_python_source

    survivors = _survivors(_corrupt(recipe_name))
    if survivors is None:
        return
    for source in survivors:
        assert validate_python_source(source).compile_ok, (
            f"{recipe_name}: survivor does not compile: {source!r}"
        )


# --- the partition covers every recipe, with no overlap --------------


def test_recipe_partition_covers_all_recipes() -> None:
    classified = set(RECOVERABLE_RECIPES) | set(NON_RECOVERABLE_RECIPES)
    assert classified == set(RECIPES_BY_NAME)
    assert set(RECOVERABLE_RECIPES).isdisjoint(NON_RECOVERABLE_RECIPES)
