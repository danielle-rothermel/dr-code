"""Frozen, named corruption recipes."""

from __future__ import annotations

import random
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import Field, SerializeAsAny, model_validator

from dr_code.models import FrozenModel
from dr_code.synthetic.models import (
    CorruptedSample,
    CorruptionCoordinate,
    RecipeCoordinate,
)
from dr_code.synthetic.names import CorruptionName, FenceLangTag
from dr_code.synthetic.corruptions import REGISTRY
from dr_code.synthetic.corruptions.base import CorruptionSettings
from dr_code.trace import ComponentSetting


class CorruptionSpec(FrozenModel):
    """One registered corruption with its settings.

    ``settings`` is the registered corruption's concrete frozen model,
    resolved while the spec crosses this validation boundary.
    """

    corruption: CorruptionName
    settings: SerializeAsAny[CorruptionSettings] = Field(
        default_factory=CorruptionSettings
    )

    @model_validator(mode="before")
    @classmethod
    def resolve_settings_model(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        corruption = CorruptionName(data["corruption"])
        settings_model = REGISTRY[corruption.value].Settings
        data["settings"] = settings_model.model_validate(
            data.get("settings", {})
        )
        return data


class Recipe(FrozenModel):
    """A named sequence of corruption specs.

    Recipes are pure data: they describe *what* to apply with *which*
    settings. The dataset builder is responsible for seeding RNG and
    calling each component.
    """

    name: str
    version: str
    corruptions: tuple[CorruptionSpec, ...]
    description: str = ""


def _require_registered(recipe: Recipe) -> Recipe:
    """Return the registered recipe `recipe` claims to be, or reject it."""
    registered = resolve_recipe(name=recipe.name, version=recipe.version)
    if recipe != registered:
        raise ValueError(
            "recipe does not match its registered coordinate: "
            f"{recipe.name}@{recipe.version}"
        )
    return registered


def apply_recipe(
    recipe: Recipe, source: str, rng: random.Random
) -> CorruptedSample:
    """Apply each component transform in order.

    The same `rng` is threaded through every transform, so a recipe is
    deterministic given its seed and settings.
    """
    _require_registered(recipe)

    current = source
    notes_chunks: list[str] = []
    for spec in recipe.corruptions:
        name = spec.corruption.value
        transform = REGISTRY[name](spec.settings)
        intermediate = transform.apply(current, rng)
        current = intermediate.corrupted_source
        if intermediate.notes:
            notes_chunks.append(f"{name}: {intermediate.notes}")
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


def _spec(corruption: CorruptionName, **settings: object) -> CorruptionSpec:
    """Build one recipe entry, validating settings against the registry."""
    return CorruptionSpec.model_validate(
        {"corruption": corruption, "settings": settings}
    )


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
        corruptions=(
            _spec(
                CorruptionName.ADD_CODE_FENCES,
                language_tag=FenceLangTag.PYTHON,
            ),
        ),
        description="Wrap source in a ```python fence.",
    ),
    Recipe(
        name="fenced_untagged",
        version="0",
        corruptions=(
            _spec(
                CorruptionName.ADD_CODE_FENCES,
                language_tag=FenceLangTag.NONE,
            ),
        ),
        description="Wrap source in an untagged ``` fence.",
    ),
    Recipe(
        name="fenced_with_prose",
        version="0",
        corruptions=(
            _spec(
                CorruptionName.ADD_CODE_FENCES,
                language_tag=FenceLangTag.PYTHON,
            ),
            _spec(CorruptionName.ADD_PROSE_WRAPPER),
        ),
        description="Fence the code, then wrap with explanatory prose.",
    ),
    Recipe(
        name="chat_indented",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_INDENTATION),),
        description="Uniformly indent the source as if from a chat quote.",
    ),
    Recipe(
        name="smart_quoted",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_SMART_QUOTES),),
        description="Replace ASCII quotes with Unicode smart quotes.",
    ),
    Recipe(
        name="crlf_tabs",
        version="0",
        corruptions=(
            _spec(CorruptionName.ADD_CRLF),
            _spec(CorruptionName.ADD_TABS),
        ),
        description="CRLF line endings combined with tab indentation.",
    ),
    Recipe(
        name="truncated_midfn",
        version="0",
        corruptions=(_spec(CorruptionName.TRUNCATE),),
        description="Truncate mid-function (mode chosen by RNG).",
    ),
    Recipe(
        name="missing_np_import",
        version="0",
        corruptions=(_spec(CorruptionName.REMOVE_IMPORTS),),
        description="Drop top-level imports (numpy in particular).",
    ),
    Recipe(
        name="mangled_import_paren",
        version="0",
        corruptions=(_spec(CorruptionName.MANGLE_IMPORT_LINES),),
        description="Syntactically mangle import lines.",
    ),
    Recipe(
        name="two_solutions",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_MULTIPLE_SOLUTIONS),),
        description="Concatenate an alternate solution after the canonical one.",
    ),
    Recipe(
        name="markdown_blockquote",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_MARKDOWN_WRAPPERS),),
        description="Wrap each line with a Markdown wrapper (e.g. blockquote).",
    ),
    Recipe(
        name="unicode_fullwidth",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_UNICODE_NOISE),),
        description="Inject benign Unicode look-alike characters.",
    ),
    Recipe(
        name="kitchen_sink",
        version="0",
        corruptions=(
            _spec(CorruptionName.ADD_SMART_QUOTES),
            _spec(CorruptionName.ADD_INDENTATION),
            _spec(CorruptionName.ADD_CRLF),
            _spec(
                CorruptionName.ADD_CODE_FENCES,
                language_tag=FenceLangTag.PY,
            ),
            _spec(CorruptionName.ADD_PROSE_WRAPPER),
        ),
        description="Multi-corruption stress test.",
    ),
    Recipe(
        name="truncated_and_unfenced",
        version="0",
        corruptions=(
            _spec(CorruptionName.TRUNCATE),
            _spec(
                CorruptionName.ADD_CODE_FENCES,
                language_tag=FenceLangTag.NONE,
            ),
        ),
        description="Truncate then wrap with an untagged fence.",
    ),
    # --- Extended set: each remaining transform exercised in isolation ---
    Recipe(
        name="trailing_whitespace",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_TRAILING_WHITESPACE),),
        description="Append trailing spaces / tabs to lines.",
    ),
    Recipe(
        name="blank_lines_noise",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_BLANK_LINES),),
        description="Inject random blank lines throughout.",
    ),
    Recipe(
        name="inline_backticks",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_INLINE_BACKTICKS),),
        description="Wrap individual identifiers in inline `code` ticks.",
    ),
    Recipe(
        name="duplicated_imports",
        version="0",
        corruptions=(_spec(CorruptionName.DUPLICATE_IMPORTS),),
        description="Duplicate top-of-file imports.",
    ),
    Recipe(
        name="comments_noise",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_COMMENTS_NOISE),),
        description="Inject incidental comments throughout.",
    ),
    Recipe(
        name="dead_code",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_DEAD_CODE),),
        description="Inject unreachable / unused statements.",
    ),
    Recipe(
        name="quote_style_swap",
        version="0",
        corruptions=(_spec(CorruptionName.CHANGE_QUOTE_STYLE),),
        description="Flip single quotes to double (or vice versa).",
    ),
    Recipe(
        name="string_form_swap",
        version="0",
        corruptions=(_spec(CorruptionName.CHANGE_STRING_FORM),),
        description="Swap between f-string / concat / format equivalents.",
    ),
    Recipe(
        name="extra_type_annotations",
        version="0",
        corruptions=(_spec(CorruptionName.ADD_TYPE_ANNOTATIONS),),
        description="Add (sometimes incorrect) type annotations.",
    ),
    Recipe(
        name="renamed_locals",
        version="0",
        corruptions=(_spec(CorruptionName.RENAME_LOCALS),),
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
    _require_registered(recipe)
    return RecipeCoordinate(
        recipe_name=recipe.name,
        version=recipe.version,
        corruptions=tuple(
            CorruptionCoordinate(
                registered_name=spec.corruption.value,
                version=REGISTRY[spec.corruption.value].VERSION,
                settings=_coordinate_settings(spec.settings),
            )
            for spec in recipe.corruptions
        ),
    )


def _coordinate_settings(
    settings: CorruptionSettings,
) -> tuple[ComponentSetting, ...]:
    """Project typed corruption settings into the bounded persisted shape."""
    entries: list[ComponentSetting] = []
    for name, value in settings.model_dump(mode="json").items():
        if not isinstance(value, str | int | float | bool | type(None)):
            raise TypeError(
                f"unsupported persisted setting shape for {name!r}: "
                f"{type(value).__name__}"
            )
        entries.append(ComponentSetting(name=name, value=value))
    return tuple(entries)


__all__ = [
    "RECIPES",
    "RECIPES_BY_NAME",
    "CorruptionSpec",
    "Recipe",
    "apply_recipe",
    "recipe_coordinate",
    "resolve_recipe",
]
