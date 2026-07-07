"""Frozen, named corruption recipes.

A recipe composes one or more inverse transforms by name. The dataset
builder applies them in order (left-to-right). The expected recovery step
set is the union of the steps from each component transform.

Recipes are an auditable, version-pinned contract: adding a recipe is
fine; changing an existing recipe's behavior must bump `DATASET_VERSION`
in `code_eval.names`.

See `docs/TESTING.md` for the synthetic corpus contract.
"""

from __future__ import annotations

import random
from typing import Final

from code_eval.models.base import FrozenModel
from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import InverseTransformName
from code_eval.synthetic.inverse_transforms import REGISTRY


class Recipe(FrozenModel):
    """A named sequence of inverse transforms.

    Recipes are pure data: they describe *what* to apply. The dataset
    builder is responsible for seeding RNG and calling each component.
    """

    name: str
    transforms: tuple[InverseTransformName, ...]
    description: str = ""


def apply_recipe(recipe: Recipe, source: str, rng: random.Random) -> CorruptedSample:
    """Apply each component transform in order, accumulating recovery steps.

    The same `rng` is threaded through every transform, so a recipe is
    deterministic given its seed.
    """
    current = source
    accumulated_steps: set[str] = set()
    notes_chunks: list[str] = []
    for tname in recipe.transforms:
        transform_cls = REGISTRY[tname.value]
        transform = transform_cls()
        intermediate = transform.apply(current, rng)
        current = intermediate.corrupted_source
        accumulated_steps |= set(intermediate.expected_recovery_steps)
        if intermediate.notes:
            notes_chunks.append(f"{tname.value}: {intermediate.notes}")
    return CorruptedSample(
        corrupted_source=current,
        expected_recovery_steps=frozenset(accumulated_steps),
        notes=" | ".join(notes_chunks),
    )


# ---------------------------------------------------------------------------
# Frozen recipe set.
#
# Order keeps the original recipe set first, with extended recipes added
# below to exercise every remaining inverse transform at least once.
# ---------------------------------------------------------------------------

RECIPES: Final[tuple[Recipe, ...]] = (
    Recipe(
        name="clean",
        transforms=(),
        description="No corruption — sanity baseline.",
    ),
    Recipe(
        name="fenced_tagged",
        transforms=(InverseTransformName.ADD_CODE_FENCES,),
        description="Wrap source in a tagged ``` fence.",
    ),
    Recipe(
        name="fenced_untagged",
        transforms=(InverseTransformName.ADD_CODE_FENCES,),
        description="Wrap source in an untagged ``` fence "
        "(transform may emit either tag — RNG-controlled).",
    ),
    Recipe(
        name="fenced_with_prose",
        transforms=(
            InverseTransformName.ADD_CODE_FENCES,
            InverseTransformName.ADD_PROSE_WRAPPER,
        ),
        description="Fence the code, then wrap with explanatory prose.",
    ),
    Recipe(
        name="chat_indented",
        transforms=(InverseTransformName.ADD_INDENTATION,),
        description="Uniformly indent the source as if from a chat quote.",
    ),
    Recipe(
        name="smart_quoted",
        transforms=(InverseTransformName.ADD_SMART_QUOTES,),
        description="Replace ASCII quotes with Unicode smart quotes.",
    ),
    Recipe(
        name="crlf_tabs",
        transforms=(
            InverseTransformName.ADD_CRLF,
            InverseTransformName.ADD_TABS,
        ),
        description="CRLF line endings combined with tab indentation.",
    ),
    Recipe(
        name="truncated_midfn",
        transforms=(InverseTransformName.TRUNCATE,),
        description="Truncate mid-function (mode chosen by RNG).",
    ),
    Recipe(
        name="missing_np_import",
        transforms=(InverseTransformName.REMOVE_IMPORTS,),
        description="Drop top-level imports (numpy in particular).",
    ),
    Recipe(
        name="mangled_import_paren",
        transforms=(InverseTransformName.MANGLE_IMPORT_LINES,),
        description="Syntactically mangle import lines.",
    ),
    Recipe(
        name="two_solutions",
        transforms=(InverseTransformName.ADD_MULTIPLE_SOLUTIONS,),
        description="Concatenate an alternate solution after the canonical one.",
    ),
    Recipe(
        name="markdown_blockquote",
        transforms=(InverseTransformName.ADD_MARKDOWN_WRAPPERS,),
        description="Wrap each line with a Markdown wrapper (e.g. blockquote).",
    ),
    Recipe(
        name="unicode_fullwidth",
        transforms=(InverseTransformName.ADD_UNICODE_NOISE,),
        description="Inject benign Unicode look-alike characters.",
    ),
    Recipe(
        name="kitchen_sink",
        transforms=(
            InverseTransformName.ADD_SMART_QUOTES,
            InverseTransformName.ADD_INDENTATION,
            InverseTransformName.ADD_CRLF,
            InverseTransformName.ADD_CODE_FENCES,
            InverseTransformName.ADD_PROSE_WRAPPER,
        ),
        description="Multi-corruption stress test.",
    ),
    Recipe(
        name="truncated_and_unfenced",
        transforms=(
            InverseTransformName.TRUNCATE,
            InverseTransformName.ADD_CODE_FENCES,
        ),
        description="Truncate then wrap with a fence.",
    ),
    # --- Extended set: each remaining transform exercised in isolation ---
    Recipe(
        name="trailing_whitespace",
        transforms=(InverseTransformName.ADD_TRAILING_WHITESPACE,),
        description="Append trailing spaces / tabs to lines.",
    ),
    Recipe(
        name="blank_lines_noise",
        transforms=(InverseTransformName.ADD_BLANK_LINES,),
        description="Inject random blank lines throughout.",
    ),
    Recipe(
        name="inline_backticks",
        transforms=(InverseTransformName.ADD_INLINE_BACKTICKS,),
        description="Wrap individual identifiers in inline `code` ticks.",
    ),
    Recipe(
        name="duplicated_imports",
        transforms=(InverseTransformName.DUPLICATE_IMPORTS,),
        description="Duplicate top-of-file imports.",
    ),
    Recipe(
        name="comments_noise",
        transforms=(InverseTransformName.ADD_COMMENTS_NOISE,),
        description="Inject incidental comments throughout.",
    ),
    Recipe(
        name="dead_code",
        transforms=(InverseTransformName.ADD_DEAD_CODE,),
        description="Inject unreachable / unused statements.",
    ),
    Recipe(
        name="quote_style_swap",
        transforms=(InverseTransformName.CHANGE_QUOTE_STYLE,),
        description="Flip single quotes to double (or vice versa).",
    ),
    Recipe(
        name="string_form_swap",
        transforms=(InverseTransformName.CHANGE_STRING_FORM,),
        description="Swap between f-string / concat / format equivalents.",
    ),
    Recipe(
        name="extra_type_annotations",
        transforms=(InverseTransformName.ADD_TYPE_ANNOTATIONS,),
        description="Add (sometimes incorrect) type annotations.",
    ),
    Recipe(
        name="renamed_locals",
        transforms=(InverseTransformName.RENAME_LOCALS,),
        description="Rename local variables to unrelated identifiers.",
    ),
)


# Convenient lookup by name.
RECIPES_BY_NAME: Final[dict[str, Recipe]] = {r.name: r for r in RECIPES}


__all__ = ["RECIPES", "RECIPES_BY_NAME", "Recipe", "apply_recipe"]
