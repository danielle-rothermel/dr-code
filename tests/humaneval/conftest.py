"""Make HumanEval test builders importable and provide shared fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dr_code.core.execution.sandbox import (
    SandboxCompletedProcess,
    SandboxRunner,
    SandboxTimeoutError,
)

_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture
def local_runner() -> SandboxRunner:
    """A real, injectable runner that keeps primitive tests fast.

    It runs the candidate under the host interpreter instead of the OCI
    sandbox; the container contract has its own probes in ``test_sandbox``.
    Injected via ``run_in_sandbox=`` rather than patched, so the seam is a
    real function argument.
    """

    def run_local_python(
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                input=input_json,
                capture_output=True,
                check=False,
                encoding="utf-8",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(str(exc)) from exc
        return SandboxCompletedProcess(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return run_local_python
