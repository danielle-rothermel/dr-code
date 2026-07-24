"""Corruption round-trips over the frozen recipe set.

Applying each ``dr_code.synthetic`` recipe to clean HumanEval-shaped code and
running the best-effort v2 definition partitions the recipes empirically
(verified across seeds 0, 1, 7, 42, 1234):

* ``RECOVERABLE`` — formatting / noise / import / wrapper pathologies the
  pipeline undoes; the recovered code is *equivalent* to the original
  (``dr_code.code_analysis.equivalent``) at every seed.

* ``NON_RECOVERABLE`` — the corruption makes a semantic or structural change
  no extraction pipeline can undo, so the recovered source is never
  equivalent to ground truth:

  - ``inline_backticks`` destroys all code-like structure — no candidate
    survives, so the run yields ``Absent`` (the pipeline gives up rather than
    fabricate code).
  - ``dead_code`` / ``renamed_locals`` / ``string_form_swap`` rewrite the AST
    (extra statements, renamed bindings, swapped string forms) — an extracted
    candidate compiles but is not semantically equivalent to the original.
    These four are the independently-verified, ratified non-recoverable set.
  - ``truncated_midfn`` / ``truncated_and_unfenced`` break the source at an
    RNG-chosen point, so the outcome is seed-dependent and never reliably
    equivalent.

  Exempting these from the *equivalence* assertion is correct behaviour, not a
  gap. For every exempted recipe we still assert that preprocessing either
  gives up or returns genuinely non-equivalent code.
"""

from __future__ import annotations

import random

import pytest

from dr_code.code_analysis import equivalent
from dr_code.preprocessing import (
    resolve_preprocessing_definition,
    run_preprocessing,
)
from dr_code.synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
from dr_code.trace import CodeArtifact, TextArtifact, is_absent

BEST_EFFORT_ID = "humaneval-best-effort"
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

#: Recipes whose corruption the best-effort v2 definition undoes, recovering
#: code equivalent to the original at every tested seed.
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

#: Recipes that make an unrecoverable semantic/structural change (see module
#: docstring). Exempted from equivalence; asserted for a non-equivalent
#: preprocessing outcome.
NON_RECOVERABLE_RECIPES = [
    "inline_backticks",
    "dead_code",
    "renamed_locals",
    "string_form_swap",
    "truncated_midfn",
    "truncated_and_unfenced",
]


def _corrupt(recipe_name: str) -> str:
    recipe = RECIPES_BY_NAME[recipe_name]
    return apply_recipe(recipe, CLEAN, random.Random(SEED)).corrupted_source


def _best_effort_v2():
    return resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )


def _run_best_effort(raw: str):
    return run_preprocessing(
        _best_effort_v2().materialize(), TextArtifact(text=raw)
    ).value("output")


# --- recoverable recipes round-trip to equivalent code ---------------


@pytest.mark.parametrize("recipe_name", RECOVERABLE_RECIPES)
def test_best_effort_recovers_corruption_to_equivalent(
    recipe_name: str,
) -> None:
    recovered = _run_best_effort(_corrupt(recipe_name))
    assert isinstance(recovered, CodeArtifact), (
        f"{recipe_name}: expected recovered CodeArtifact, got {recovered!r}"
    )
    assert equivalent(CLEAN, recovered.source), (
        f"{recipe_name}: recovered code is not equivalent to the original"
    )


# --- non-recoverable recipes: exempt from equivalence, assert outcome -


@pytest.mark.parametrize("recipe_name", NON_RECOVERABLE_RECIPES)
def test_non_recoverable_does_not_recover_equivalent_code(
    recipe_name: str,
) -> None:
    preprocessing = _run_best_effort(_corrupt(recipe_name))
    source = None if is_absent(preprocessing) else preprocessing.source
    if source is not None:
        assert not equivalent(CLEAN, source), (
            f"{recipe_name}: unexpectedly recovered equivalent code; "
            f"it should be reclassified as recoverable"
        )


def test_inline_backticks_yields_absent() -> None:
    # The strongest give-up case: inline backticks destroy all code-like
    # structure, so no candidate survives and the run is Absent.
    out = _run_best_effort(_corrupt("inline_backticks"))
    assert is_absent(out)


# --- the partition covers every recipe, with no overlap --------------


def test_recipe_partition_covers_all_recipes() -> None:
    classified = set(RECOVERABLE_RECIPES) | set(NON_RECOVERABLE_RECIPES)
    assert classified == set(RECIPES_BY_NAME)
    assert set(RECOVERABLE_RECIPES).isdisjoint(NON_RECOVERABLE_RECIPES)
