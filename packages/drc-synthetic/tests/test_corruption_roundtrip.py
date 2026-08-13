from __future__ import annotations

import random

import pytest

from dr_code.core.source.python_analysis import equivalent
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_preprocessing,
)
from drc_synthetic.corruption_recipes import RECIPES_BY_NAME, apply_recipe
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
    output = _RUNNER.run(TextArtifact(text=raw)).value(OUTPUT_KEY)
    if is_absent(output):
        return None
    assert isinstance(output, InspectedCodeCandidateSetArtifact)
    return tuple(item.candidate.source for item in output.candidates)


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


@pytest.mark.parametrize("recipe_name", NON_RECOVERABLE_RECIPES)
def test_non_recoverable_corruption_recovers_nothing_equivalent(
    recipe_name: str,
) -> None:
    survivors = _survivors(_corrupt(recipe_name))
    if survivors is None:
        return
    assert not any(equivalent(CLEAN, source) for source in survivors), (
        f"{recipe_name}: unexpectedly recovered an equivalent candidate; "
        f"it should be reclassified as recoverable"
    )


def test_inline_backticks_yields_absent() -> None:
    assert _survivors(_corrupt("inline_backticks")) is None


@pytest.mark.parametrize(
    "recipe_name", RECOVERABLE_RECIPES + NON_RECOVERABLE_RECIPES
)
def test_every_survivor_compiles(recipe_name: str) -> None:
    from dr_code.core.source.python_analysis import validate_python_source

    survivors = _survivors(_corrupt(recipe_name))
    if survivors is None:
        return
    for source in survivors:
        assert validate_python_source(source).compile_ok, (
            f"{recipe_name}: survivor does not compile: {source!r}"
        )


def test_recipe_partition_covers_all_recipes() -> None:
    classified = set(RECOVERABLE_RECIPES) | set(NON_RECOVERABLE_RECIPES)
    assert classified == set(RECIPES_BY_NAME)
    assert set(RECOVERABLE_RECIPES).isdisjoint(NON_RECOVERABLE_RECIPES)
