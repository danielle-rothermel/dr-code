"""Frozen, named corruption recipes."""

from __future__ import annotations

import random
from types import MappingProxyType
from typing import Final

from dr_code.models import FrozenModel
from dr_code.synthetic.models import (
    CorruptedSample,
    CorruptionCoordinate,
    RecipeCoordinate,
)
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions import REGISTRY


class Recipe(FrozenModel):
    """A named sequence of corruptions.

    Recipes are pure data: they describe *what* to apply. The dataset
    builder is responsible for seeding RNG and calling each component.
    """

    name: str
    version: str
    corruptions: tuple[CorruptionName, ...]
    description: str = ""


def apply_recipe(
    recipe: Recipe, source: str, rng: random.Random
) -> CorruptedSample:
    """Apply each component transform in order.

    The same `rng` is threaded through every transform, so a recipe is
    deterministic given its seed.
    """
    registered = resolve_recipe(name=recipe.name, version=recipe.version)
    if recipe != registered:
        raise ValueError(
            "recipe does not match its registered coordinate: "
            f"{recipe.name}@{recipe.version}"
        )

    current = source
    notes_chunks: list[str] = []
    for tname in recipe.corruptions:
        transform_cls = REGISTRY[tname.value]
        transform = transform_cls()
        intermediate = transform.apply(current, rng)
        current = intermediate.corrupted_source
        if intermediate.notes:
            notes_chunks.append(f"{tname.value}: {intermediate.notes}")
    return CorruptedSample(
        corrupted_source=current,
        notes=" | ".join(notes_chunks),
    )


# ---------------------------------------------------------------------------
# Frozen recipe set.
#
# Order keeps the original recipe set first, with extended recipes added
# below to exercise every remaining corruption at least once.
# ---------------------------------------------------------------------------

RECIPES: Final[tuple[Recipe, ...]] = (
    Recipe(
        name="clean",
        version="0",
        corruptions=(),
        description="No corruption — sanity baseline.",
    ),
    Recipe(
        name="fenced_tagged",
        version="0",
        corruptions=(CorruptionName.ADD_CODE_FENCES,),
        description="Wrap source in a tagged ``` fence.",
    ),
    Recipe(
        name="fenced_untagged",
        version="0",
        corruptions=(CorruptionName.ADD_CODE_FENCES,),
        description="Wrap source in an untagged ``` fence "
        "(transform may emit either tag — RNG-controlled).",
    ),
    Recipe(
        name="fenced_with_prose",
        version="0",
        corruptions=(
            CorruptionName.ADD_CODE_FENCES,
            CorruptionName.ADD_PROSE_WRAPPER,
        ),
        description="Fence the code, then wrap with explanatory prose.",
    ),
    Recipe(
        name="chat_indented",
        version="0",
        corruptions=(CorruptionName.ADD_INDENTATION,),
        description="Uniformly indent the source as if from a chat quote.",
    ),
    Recipe(
        name="smart_quoted",
        version="0",
        corruptions=(CorruptionName.ADD_SMART_QUOTES,),
        description="Replace ASCII quotes with Unicode smart quotes.",
    ),
    Recipe(
        name="crlf_tabs",
        version="0",
        corruptions=(
            CorruptionName.ADD_CRLF,
            CorruptionName.ADD_TABS,
        ),
        description="CRLF line endings combined with tab indentation.",
    ),
    Recipe(
        name="truncated_midfn",
        version="0",
        corruptions=(CorruptionName.TRUNCATE,),
        description="Truncate mid-function (mode chosen by RNG).",
    ),
    Recipe(
        name="missing_np_import",
        version="0",
        corruptions=(CorruptionName.REMOVE_IMPORTS,),
        description="Drop top-level imports (numpy in particular).",
    ),
    Recipe(
        name="mangled_import_paren",
        version="0",
        corruptions=(CorruptionName.MANGLE_IMPORT_LINES,),
        description="Syntactically mangle import lines.",
    ),
    Recipe(
        name="two_solutions",
        version="0",
        corruptions=(CorruptionName.ADD_MULTIPLE_SOLUTIONS,),
        description="Concatenate an alternate solution after the canonical one.",
    ),
    Recipe(
        name="markdown_blockquote",
        version="0",
        corruptions=(CorruptionName.ADD_MARKDOWN_WRAPPERS,),
        description="Wrap each line with a Markdown wrapper (e.g. blockquote).",
    ),
    Recipe(
        name="unicode_fullwidth",
        version="0",
        corruptions=(CorruptionName.ADD_UNICODE_NOISE,),
        description="Inject benign Unicode look-alike characters.",
    ),
    Recipe(
        name="kitchen_sink",
        version="0",
        corruptions=(
            CorruptionName.ADD_SMART_QUOTES,
            CorruptionName.ADD_INDENTATION,
            CorruptionName.ADD_CRLF,
            CorruptionName.ADD_CODE_FENCES,
            CorruptionName.ADD_PROSE_WRAPPER,
        ),
        description="Multi-corruption stress test.",
    ),
    Recipe(
        name="truncated_and_unfenced",
        version="0",
        corruptions=(
            CorruptionName.TRUNCATE,
            CorruptionName.ADD_CODE_FENCES,
        ),
        description="Truncate then wrap with a fence.",
    ),
    # --- Extended set: each remaining transform exercised in isolation ---
    Recipe(
        name="trailing_whitespace",
        version="0",
        corruptions=(CorruptionName.ADD_TRAILING_WHITESPACE,),
        description="Append trailing spaces / tabs to lines.",
    ),
    Recipe(
        name="blank_lines_noise",
        version="0",
        corruptions=(CorruptionName.ADD_BLANK_LINES,),
        description="Inject random blank lines throughout.",
    ),
    Recipe(
        name="inline_backticks",
        version="0",
        corruptions=(CorruptionName.ADD_INLINE_BACKTICKS,),
        description="Wrap individual identifiers in inline `code` ticks.",
    ),
    Recipe(
        name="duplicated_imports",
        version="0",
        corruptions=(CorruptionName.DUPLICATE_IMPORTS,),
        description="Duplicate top-of-file imports.",
    ),
    Recipe(
        name="comments_noise",
        version="0",
        corruptions=(CorruptionName.ADD_COMMENTS_NOISE,),
        description="Inject incidental comments throughout.",
    ),
    Recipe(
        name="dead_code",
        version="0",
        corruptions=(CorruptionName.ADD_DEAD_CODE,),
        description="Inject unreachable / unused statements.",
    ),
    Recipe(
        name="quote_style_swap",
        version="0",
        corruptions=(CorruptionName.CHANGE_QUOTE_STYLE,),
        description="Flip single quotes to double (or vice versa).",
    ),
    Recipe(
        name="string_form_swap",
        version="0",
        corruptions=(CorruptionName.CHANGE_STRING_FORM,),
        description="Swap between f-string / concat / format equivalents.",
    ),
    Recipe(
        name="extra_type_annotations",
        version="0",
        corruptions=(CorruptionName.ADD_TYPE_ANNOTATIONS,),
        description="Add (sometimes incorrect) type annotations.",
    ),
    Recipe(
        name="renamed_locals",
        version="0",
        corruptions=(CorruptionName.RENAME_LOCALS,),
        description="Rename local variables to unrelated identifiers.",
    ),
)


# Convenient lookup by name.
RECIPES_BY_NAME: Final = MappingProxyType({r.name: r for r in RECIPES})


def resolve_recipe(*, name: str, version: str) -> Recipe:
    recipe = RECIPES_BY_NAME.get(name)
    if recipe is None or recipe.version != version:
        raise ValueError(f"unsupported synthetic recipe: {name}@{version}")
    return recipe


def recipe_coordinate(recipe: Recipe) -> RecipeCoordinate:
    registered = resolve_recipe(name=recipe.name, version=recipe.version)
    if recipe != registered:
        raise ValueError(
            "recipe does not match its registered coordinate: "
            f"{recipe.name}@{recipe.version}"
        )
    return RecipeCoordinate(
        recipe_name=recipe.name,
        version=recipe.version,
        corruptions=tuple(
            CorruptionCoordinate(
                registered_name=name.value,
                version=REGISTRY[name.value].VERSION,
            )
            for name in recipe.corruptions
        ),
    )


__all__ = [
    "RECIPES",
    "RECIPES_BY_NAME",
    "Recipe",
    "apply_recipe",
    "recipe_coordinate",
    "resolve_recipe",
]
