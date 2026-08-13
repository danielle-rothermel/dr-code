from __future__ import annotations

import random
from typing import Final

import pytest
from pydantic import ValidationError

from drc_synthetic.corruption_recipes import (
    RECIPES,
    RECIPES_BY_NAME,
    CorruptionSpec,
    Recipe,
    apply_recipe,
    recipe_coordinate,
    resolve_recipe,
)
from drc_synthetic.corruptions.add_code_fences import AddCodeFencesSettings
from drc_synthetic.corruptions.base import CorruptionSettings
from drc_synthetic.models import CorruptionCoordinate
from drc_synthetic.names import CorruptionName, FenceLangTag

SOURCE = "def add(a, b):\n    return a + b\n"


def _impersonating_recipe() -> Recipe:
    registered = RECIPES_BY_NAME["clean"]
    return registered.model_copy(
        update={
            "corruptions": (
                CorruptionSpec.model_validate(
                    {"corruption": CorruptionName.ADD_CRLF}
                ),
            )
        }
    )


def test_apply_recipe_rejects_impersonating_recipe() -> None:
    with pytest.raises(
        ValueError, match="does not match its registered coordinate"
    ):
        apply_recipe(_impersonating_recipe(), SOURCE, random.Random(0))


def test_recipe_coordinate_rejects_impersonating_recipe() -> None:
    with pytest.raises(
        ValueError, match="does not match its registered coordinate"
    ):
        recipe_coordinate(_impersonating_recipe())


def test_apply_recipe_rejects_altered_settings() -> None:
    registered = RECIPES_BY_NAME["fenced_tagged"]
    retagged = registered.model_copy(
        update={
            "corruptions": (
                CorruptionSpec.model_validate(
                    {
                        "corruption": CorruptionName.ADD_CODE_FENCES,
                        "settings": {"language_tag": FenceLangTag.PYTHON3},
                    }
                ),
            )
        }
    )

    with pytest.raises(
        ValueError, match="does not match its registered coordinate"
    ):
        apply_recipe(retagged, SOURCE, random.Random(0))


@pytest.mark.parametrize(
    "name, version",
    [
        ("no_such_recipe", "0"),
        ("clean", "1"),
        ("no_such_recipe", "9"),
    ],
)
def test_resolve_recipe_rejects_unregistered_coordinate(
    name: str, version: str
) -> None:
    with pytest.raises(ValueError, match="unsupported synthetic recipe"):
        resolve_recipe(name=name, version=version)


def test_fence_recipes_differ_by_settings_not_only_by_name() -> None:
    # These literal coordinates pin persisted recipe identity.
    tagged = recipe_coordinate(RECIPES_BY_NAME["fenced_tagged"])
    untagged = recipe_coordinate(RECIPES_BY_NAME["fenced_untagged"])

    assert tagged.corruptions != untagged.corruptions
    assert tagged.corruptions == (
        CorruptionCoordinate(
            registered_name="add_code_fences",
            version="0",
            settings=({"name": "language_tag", "value": "python"},),
        ),
    )
    assert untagged.corruptions == (
        CorruptionCoordinate(
            registered_name="add_code_fences",
            version="0",
            settings=({"name": "language_tag", "value": ""},),
        ),
    )


def test_fence_recipes_produce_their_declared_tag() -> None:
    tagged = apply_recipe(
        RECIPES_BY_NAME["fenced_tagged"], SOURCE, random.Random(0)
    )
    untagged = apply_recipe(
        RECIPES_BY_NAME["fenced_untagged"], SOURCE, random.Random(0)
    )

    assert tagged.corrupted_source.startswith("```python\n")
    assert untagged.corrupted_source.startswith("```\n")


def test_recipe_coordinate_gives_settingless_corruptions_an_empty_tuple() -> (
    None
):
    # Empty settings are part of the persisted corruption coordinate.
    coordinate = recipe_coordinate(RECIPES_BY_NAME["smart_quoted"])

    assert coordinate.corruptions == (
        CorruptionCoordinate(
            registered_name="add_smart_quotes", version="0", settings=()
        ),
    )


# This literal pins every settings-bearing persisted recipe coordinate.
_SETTINGS_BEARING_RECIPE_COORDINATES: Final[
    dict[str, tuple[dict[str, object], ...]]
] = {
    "fenced_tagged": (
        {
            "registered_name": "add_code_fences",
            "version": "0",
            "settings": ({"name": "language_tag", "value": "python"},),
        },
    ),
    "fenced_untagged": (
        {
            "registered_name": "add_code_fences",
            "version": "0",
            "settings": ({"name": "language_tag", "value": ""},),
        },
    ),
    "fenced_with_prose": (
        {
            "registered_name": "add_code_fences",
            "version": "0",
            "settings": ({"name": "language_tag", "value": "python"},),
        },
        {
            "registered_name": "add_prose_wrapper",
            "version": "0",
            "settings": (),
        },
    ),
    "kitchen_sink": (
        {
            "registered_name": "add_smart_quotes",
            "version": "0",
            "settings": (),
        },
        {
            "registered_name": "add_indentation",
            "version": "0",
            "settings": (),
        },
        {"registered_name": "add_crlf", "version": "0", "settings": ()},
        {
            "registered_name": "add_code_fences",
            "version": "0",
            "settings": ({"name": "language_tag", "value": "py"},),
        },
        {
            "registered_name": "add_prose_wrapper",
            "version": "0",
            "settings": (),
        },
    ),
    "truncated_and_unfenced": (
        {"registered_name": "truncate", "version": "0", "settings": ()},
        {
            "registered_name": "add_code_fences",
            "version": "0",
            "settings": ({"name": "language_tag", "value": ""},),
        },
    ),
}


def test_settings_bearing_recipe_coordinates_are_pinned() -> None:
    actual = {
        recipe.name: tuple(
            corruption.model_dump(mode="python")
            for corruption in recipe_coordinate(recipe).corruptions
        )
        for recipe in RECIPES
        if any(
            corruption.settings
            for corruption in recipe_coordinate(recipe).corruptions
        )
    }

    assert actual == _SETTINGS_BEARING_RECIPE_COORDINATES


def test_corruption_spec_rejects_settings_from_another_corruption() -> None:
    class _ForeignSettings(CorruptionSettings):
        bogus: int = 7

    with pytest.raises(ValidationError):
        CorruptionSpec.model_validate(
            {
                "corruption": CorruptionName.ADD_SMART_QUOTES,
                "settings": _ForeignSettings(),
            }
        )


def test_corruption_spec_accepts_the_named_corruptions_settings_instance() -> (
    None
):
    spec = CorruptionSpec.model_validate(
        {
            "corruption": CorruptionName.ADD_CODE_FENCES,
            "settings": AddCodeFencesSettings(
                language_tag=FenceLangTag.PYTHON
            ),
        }
    )
    assert spec.settings == AddCodeFencesSettings(
        language_tag=FenceLangTag.PYTHON
    )


def test_corruption_spec_accepts_plain_dict_settings() -> None:
    spec = CorruptionSpec.model_validate(
        {
            "corruption": CorruptionName.ADD_CODE_FENCES,
            "settings": {"language_tag": FenceLangTag.PY},
        }
    )
    assert spec.settings == AddCodeFencesSettings(language_tag=FenceLangTag.PY)


def test_corruption_spec_missing_corruption_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CorruptionSpec.model_validate({"settings": {}})
