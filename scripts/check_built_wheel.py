#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_SMOKE_PROGRAM = r"""
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import dr_code.caching
import dr_code.evaluation
import dr_code.metrics
import dr_code.preprocessing
import dr_code.synthetic
import dr_code.trace
from dr_exec import ProcessExecutor
from dr_code.core.execution.executor import (
    host_process_executor,
    run_python_source,
)
from dr_code.humaneval.runner import runner_script

expected_version = sys.argv[1]
installed_version = version("dr-code")
if installed_version != expected_version:
    raise SystemExit(
        f"installed dr-code version {installed_version!r} does not match "
        f"{expected_version!r}"
    )
if "def dr_exec_main" not in runner_script():
    raise SystemExit("installed wheel is missing the HumanEval driver resource")

driver_source = "def dr_exec_main(request, emit):\n    print('wheel-smoke')\n"
with TemporaryDirectory(prefix="dr-code-wheel-records-") as record_root:
    executor = host_process_executor(
        Path(record_root),
        runtime_executable=Path(sys.executable),
    )
    if not isinstance(executor, ProcessExecutor):
        raise SystemExit("production executor is not a ProcessExecutor")
    if sys.platform == "darwin":
        completed = run_python_source(
            executor,
            source=driver_source,
            input_json="{}",
            timeout_seconds=5.0,
        )
        if completed.returncode != 0 or completed.stdout != "wheel-smoke\n":
            raise SystemExit(f"installed-wheel execution failed: {completed!r}")

print(f"installed wheel smoke passed for dr-code {installed_version}")
"""


def _project_version() -> str:
    with (_ROOT / "pyproject.toml").open("rb") as file:
        document = tomllib.load(file)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml has no project table")
    version = project.get("version")
    if not isinstance(version, str):
        raise ValueError("pyproject.toml project version must be a string")
    return version


def _built_wheel(version: str) -> Path:
    wheels = tuple((_ROOT / "dist").glob(f"dr_code-{version}-*.whl"))
    if len(wheels) != 1:
        raise ValueError(
            f"expected exactly one built wheel for dr-code {version}, "
            f"found {len(wheels)}"
        )
    return wheels[0]


def main() -> int:
    version = _project_version()
    wheel = _built_wheel(version)
    with tempfile.TemporaryDirectory(
        prefix="dr-code-installed-wheel-"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        environment = temporary_root / "venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                sys.executable,
                str(environment),
            ],
            check=True,
        )
        environment_python = environment / "bin" / "python"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment_python),
                "--no-cache",
                str(wheel),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(environment_python),
                "-I",
                "-c",
                _SMOKE_PROGRAM,
                version,
            ],
            cwd=temporary_root,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
