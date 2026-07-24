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

  Exempting these from the positive *equivalence* assertion is correct
  behaviour, not a gap. For every exempted recipe we still assert that the
  official pipeline does not fabricate a candidate equivalent to the original.
"""

from __future__ import annotations

import random

import pytest

from dr_code.code_analysis import equivalent
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)
from dr_code.synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
from dr_code.trace import CodeCandidateSetArtifact, TextArtifact, is_absent

SEED = 0
RUNNER = bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)

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

#: Recipes that make an unrecoverable semantic or structural change.
#: Exempted from equivalence and covered by their direct corruption tests.
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


def _run_best_effort(raw: str):
    return RUNNER.run(TextArtifact(text=raw)).value("output")


# --- recoverable recipes round-trip to equivalent code ---------------


@pytest.mark.parametrize("recipe_name", RECOVERABLE_RECIPES)
def test_best_effort_recovers_corruption_to_equivalent(
    recipe_name: str,
) -> None:
    recovered = _run_best_effort(_corrupt(recipe_name))
    assert isinstance(recovered, CodeCandidateSetArtifact), (
        f"{recipe_name}: expected candidate set, got {recovered!r}"
    )
    assert any(equivalent(CLEAN, source) for source in recovered.candidates), (
        f"{recipe_name}: recovered code is not equivalent to the original"
    )


# --- non-recoverable recipes do not fabricate equivalent source -------


@pytest.mark.parametrize("recipe_name", NON_RECOVERABLE_RECIPES)
def test_non_recoverable_does_not_fabricate_equivalent_code(
    recipe_name: str,
) -> None:
    raw = _corrupt(recipe_name)
    output = _run_best_effort(raw)
    if not is_absent(output):
        assert isinstance(output, CodeCandidateSetArtifact)
        assert not any(
            equivalent(CLEAN, source) for source in output.candidates
        ), (
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
