"""Installed identity dependency and persisted-schema version contracts."""

from __future__ import annotations

import tomllib
from pathlib import Path

from dr_code.eval import METRIC_RECORD_SCHEMA_VERSION
from dr_code.trace import TRACE_SCHEMA_VERSION

_ROOT = Path(__file__).parents[1]


def test_project_version_and_identity_dependency_are_deliberate() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())

    assert project["project"]["version"] == "0.2.0"
    assert "dr-serialize==0.1.0" in project["project"]["dependencies"]
    assert METRIC_RECORD_SCHEMA_VERSION == 3
    assert TRACE_SCHEMA_VERSION == 4


def test_lock_preserves_the_exact_identity_dependency_boundary() -> None:
    lock = tomllib.loads((_ROOT / "uv.lock").read_text())
    dr_code = next(
        package for package in lock["package"] if package["name"] == "dr-code"
    )

    assert dr_code["version"] == "0.2.0"
    assert {
        requirement["specifier"]
        for requirement in dr_code["metadata"]["requires-dist"]
        if requirement["name"] == "dr-serialize"
    } == {"==0.1.0"}
