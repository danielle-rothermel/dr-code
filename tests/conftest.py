"""Shared deterministic module-runner fixture on the dr-exec executor."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
from dr_exec import (
    Budgets,
    EnvironmentGrant,
    Records,
    RunResult,
    run_tool,
)

_MODULE_TIMEOUT_SECONDS = 30.0

# COLUMNS/NO_COLOR pin CLI rendering; PYTHONHASHSEED pins hash-dependent
# ordering. The self-invocation probe is a trusted first-party tool, so it
# rides ``run_tool`` with an ambient overlay: the hermetic Python runner's
# ``-I`` would strip PYTHONHASHSEED and defeat the determinism this grant
# exists to provide. Module resolution is via the installed editable package,
# so the engine's per-run scratch cwd does not affect ``python -m`` lookup.
_MODULE_RUNNER_ENVIRONMENT = EnvironmentGrant.overlay(
    extra={
        "COLUMNS": "120",
        "NO_COLOR": "1",
        "PYTHONHASHSEED": "0",
    }
)


@dataclass(frozen=True, slots=True)
class PythonModuleRunner:
    """Run an installed module in a bounded, non-interactive interpreter."""

    timeout_seconds: float

    def __call__(self, module: str, *arguments: str) -> RunResult:
        return run_tool(
            [sys.executable, "-m", module, *arguments],
            budgets=Budgets(wall_clock=self.timeout_seconds),
            records=Records.none(),
            environment=_MODULE_RUNNER_ENVIRONMENT,
        )


@pytest.fixture
def run_python_module() -> PythonModuleRunner:
    return PythonModuleRunner(timeout_seconds=_MODULE_TIMEOUT_SECONDS)
