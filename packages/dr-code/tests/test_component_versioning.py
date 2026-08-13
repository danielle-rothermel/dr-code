from __future__ import annotations

import tomllib
from pathlib import Path

from dr_code.metrics.registry import REGISTRY as METRIC_REGISTRY
from dr_code.preprocessing.definitions import _DEFINITIONS
from dr_code.preprocessing.registry import REGISTRY as STEP_REGISTRY

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
    }
    assert len(versions) == (
        len(STEP_REGISTRY) + len(METRIC_REGISTRY) + len(_DEFINITIONS)
    )
    return versions


def test_development_mode_requires_initial_version_for_every_component() -> (
    None
):
    config = tomllib.loads(
        (_REPOSITORY_ROOT.parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["tool"]["dr-code"]["component-versioning"]

    assert config == {
        "development-mode": True,
        "initial-version": "0",
    }
    versions = _component_versions()
    assert versions
    assert set(versions.values()) == {config["initial-version"]}
