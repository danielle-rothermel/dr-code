from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_TIMEOUT_SECONDS = 30.0

# Shared plain-module test helpers (e.g. _executor_stubs) live beside this
# conftest; importlib import mode does not put the tests root on sys.path.
_TESTS_ROOT = str(Path(__file__).resolve().parent)
if _TESTS_ROOT not in sys.path:
    sys.path.insert(0, _TESTS_ROOT)

import _candidate_job_builders  # noqa: F401,E402  register stub operators


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
    cwd: Path
    timeout_seconds: float

    def __call__(
        self, script_path: Path, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        # Isolated mode removes test paths, so invoke scripts by path, not -m.
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
