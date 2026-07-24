"""Corruption round-trips over the frozen recipe set.

Applying each ``dr_code.synthetic`` recipe to clean HumanEval-shaped code and
running it through the best-effort profile partitions the recipes empirically
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
  gap. For every exempted recipe we still assert something meaningful: the
  profile either gives up, or returns a candidate the pipeline actually
  produced and that is genuinely not equivalent to ground truth. No exempted
  recipe sits in an assertion-free bucket.
"""

from __future__ import annotations

import random

import pytest

from dr_code.code_analysis import equivalent
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
    extract_code_with_profile,
)
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    run_preprocessing,
)
from dr_code.synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
from dr_code.trace import (
    OUTPUT_KEY,
    CodeCandidateSetArtifact,
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

#: Recipes whose corruption the best-effort profile undoes, recovering code
#: equivalent to the original at every tested seed.
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
#: docstring). Exempted from equivalence; asserted for parser parity.
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


def _pipeline_candidates(raw: str) -> tuple[str, ...]:
    """Every candidate the shared pipeline produces, before selection."""
    output = run_preprocessing(
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION, TextArtifact(text=raw)
    ).value(OUTPUT_KEY)
    if is_absent(output):
        return ()
    assert isinstance(output, CodeCandidateSetArtifact)
    return output.candidates


def _best_effort(raw: str) -> str | None:
    return extract_code_with_profile(
        raw, profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
    ).extracted_code


# --- recoverable recipes round-trip to equivalent code ---------------


@pytest.mark.parametrize("recipe_name", RECOVERABLE_RECIPES)
def test_best_effort_recovers_corruption_to_equivalent(
    recipe_name: str,
) -> None:
    recovered = _best_effort(_corrupt(recipe_name))
    assert recovered is not None, (
        f"{recipe_name}: expected recovered code, got no candidate"
    )
    assert equivalent(CLEAN, recovered), (
        f"{recipe_name}: recovered code is not equivalent to the original"
    )


# --- non-recoverable recipes: exempt from equivalence, assert parity -


@pytest.mark.parametrize("recipe_name", NON_RECOVERABLE_RECIPES)
def test_non_recoverable_gives_up_or_returns_a_pipeline_candidate(
    recipe_name: str,
) -> None:
    # No extraction pipeline can recover equivalence here, so we assert the
    # fallback contract: the profile gives up, or it returns a candidate the
    # pipeline genuinely produced.
    raw = _corrupt(recipe_name)
    extracted = _best_effort(raw)
    candidates = _pipeline_candidates(raw)
    if extracted is None:
        return
    assert extracted in candidates, (
        f"{recipe_name}: selection returned source the pipeline never produced"
    )
    # And that code is genuinely not equivalent — confirming the exemption
    # is warranted, not a masked recoverable case.
    assert not equivalent(CLEAN, extracted), (
        f"{recipe_name}: unexpectedly recovered equivalent code; "
        f"it should be reclassified as recoverable"
    )


def test_inline_backticks_yields_no_candidate() -> None:
    # The strongest give-up case: inline backticks destroy all code-like
    # structure, so no candidate survives at all.
    raw = _corrupt("inline_backticks")
    assert _pipeline_candidates(raw) == ()
    assert _best_effort(raw) is None


# --- the partition covers every recipe, with no overlap --------------


def test_recipe_partition_covers_all_recipes() -> None:
    classified = set(RECOVERABLE_RECIPES) | set(NON_RECOVERABLE_RECIPES)
    assert classified == set(RECIPES_BY_NAME)
    assert set(RECOVERABLE_RECIPES).isdisjoint(NON_RECOVERABLE_RECIPES)
