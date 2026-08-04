"""Registered-coordinate guards and settings-bearing recipe coordinates."""

from __future__ import annotations

import random

import pytest

from dr_code.synthetic.corruption_recipes import (
    RECIPES_BY_NAME,
    CorruptionSpec,
    Recipe,
    apply_recipe,
    recipe_coordinate,
    resolve_recipe,
)
from dr_code.synthetic.models import CorruptionCoordinate
from dr_code.synthetic.names import CorruptionName, FenceLangTag

SOURCE = "def add(a, b):\n    return a + b\n"


def _impersonating_recipe() -> Recipe:
    """A recipe claiming a registered name/version with altered corruptions."""
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
    coordinate = recipe_coordinate(RECIPES_BY_NAME["smart_quoted"])

    assert coordinate.corruptions == (
        CorruptionCoordinate(
            registered_name="add_smart_quotes", version="0", settings=()
        ),
    )
