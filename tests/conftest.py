"""Shared deterministic subprocess helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_TIMEOUT_SECONDS = 30.0


def _bounded_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "COLUMNS": "120",
            "NO_COLOR": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


@dataclass(frozen=True, slots=True)
class PythonModuleRunner:
    """Run an installed module in a bounded, non-interactive interpreter."""

    cwd: Path
    timeout_seconds: float

    def __call__(
        self, module: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", "-m", module, *arguments],
            cwd=self.cwd,
            env=_bounded_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )


@dataclass(frozen=True, slots=True)
class PythonScriptRunner:
    """Run a script by path in a bounded, non-interactive interpreter.

    Isolated mode strips ``PYTHONPATH`` and the cwd, so scripts living
    under ``tests/`` are unreachable by ``-m`` and run by path instead.
    """

    cwd: Path
    timeout_seconds: float

    def __call__(
        self, script_path: Path, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", str(script_path), *arguments],
            cwd=self.cwd,
            env=_bounded_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )


@pytest.fixture
def run_python_module() -> PythonModuleRunner:
    return PythonModuleRunner(
        cwd=_REPO_ROOT,
        timeout_seconds=_MODULE_TIMEOUT_SECONDS,
    )


@pytest.fixture
def run_python_script() -> PythonScriptRunner:
    return PythonScriptRunner(
        cwd=_REPO_ROOT,
        timeout_seconds=_MODULE_TIMEOUT_SECONDS,
    )
