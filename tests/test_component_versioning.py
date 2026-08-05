"""Repository contract for production semantic component versions."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dr_code.humaneval.profiles import (
    HUMANEVAL_METRICS_PROFILE_ID,
    HUMANEVAL_METRICS_PROFILE_VERSION,
    _SCORING_PROFILES,
)
from dr_code.humaneval.task import (
    HUMANEVAL_OVERRIDE_SET_ID,
    HUMANEVAL_OVERRIDE_SET_VERSION,
)
from dr_code.metrics.registry import REGISTRY as METRIC_REGISTRY
from dr_code.preprocessing.definitions import _DEFINITIONS
from dr_code.preprocessing.registry import REGISTRY as STEP_REGISTRY
from dr_code.synthetic.corruption_recipes import RECIPES
from dr_code.synthetic.corruptions import REGISTRY as CORRUPTION_REGISTRY

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _component_versions() -> dict[str, str]:
    versions = {
        **{
            f"preprocessing-step:{name}": step.VERSION
            for name, step in STEP_REGISTRY.items()
        },
        **{
            f"metric-operator:{name}": operator.VERSION
            for name, operator in METRIC_REGISTRY.items()
        },
        **{
            f"preprocessing-definition:{definition_id}": definition.version
            for (definition_id, _version), definition in _DEFINITIONS.items()
        },
        **{
            f"scoring-profile:{profile_id}": profile.version
            for (profile_id, _version), profile in _SCORING_PROFILES.items()
        },
        **{
            f"synthetic-corruption:{name}": corruption.VERSION
            for name, corruption in CORRUPTION_REGISTRY.items()
        },
        **{
            f"synthetic-recipe:{recipe.name}": recipe.version
            for recipe in RECIPES
        },
        f"override-set:{HUMANEVAL_OVERRIDE_SET_ID}": (
            HUMANEVAL_OVERRIDE_SET_VERSION
        ),
        f"metrics-profile:{HUMANEVAL_METRICS_PROFILE_ID}": (
            HUMANEVAL_METRICS_PROFILE_VERSION
        ),
    }
    assert len(versions) == (
        len(STEP_REGISTRY)
        + len(METRIC_REGISTRY)
        + len(_DEFINITIONS)
        + len(_SCORING_PROFILES)
        + len(CORRUPTION_REGISTRY)
        + len(RECIPES)
        + 2
    )
    return versions


def test_development_mode_requires_initial_version_for_every_component() -> (
    None
):
    configuration = tomllib.loads(
        (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["dr-code"]["component-versioning"]

    assert configuration == {
        "development-mode": True,
        "initial-version": "0",
    }
    versions = _component_versions()
    assert versions
    assert set(versions.values()) == {configuration["initial-version"]}
