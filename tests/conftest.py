"""Shared pytest configuration."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest


def _docker_available() -> bool:
    if os.environ.get("DR_CODE_SKIP_DOCKER") == "1":
        return False
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "docker: requires Docker daemon and nl-code eval image",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del config
    if _docker_available():
        return
    skip_marker = pytest.mark.skip(
        reason="Docker unavailable (set DR_CODE_SKIP_DOCKER=1 to suppress)",
    )
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_marker)
