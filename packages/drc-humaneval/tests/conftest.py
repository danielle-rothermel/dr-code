from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODULE_TIMEOUT_SECONDS = 30.0

_HUMANEVAL_TESTS_ROOT = str(Path(__file__).resolve().parent)
if _HUMANEVAL_TESTS_ROOT not in sys.path:
    sys.path.insert(0, _HUMANEVAL_TESTS_ROOT)

_EVAL_TESTS_ROOT = str(Path(__file__).resolve().parent / "evaluation")
if _EVAL_TESTS_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_TESTS_ROOT)


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


@pytest.fixture
def run_python_module() -> PythonModuleRunner:
    return PythonModuleRunner(
        cwd=_REPO_ROOT,
        timeout_seconds=_MODULE_TIMEOUT_SECONDS,
    )
