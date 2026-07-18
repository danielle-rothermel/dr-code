"""Pytest fixtures for the ``dr_code.metrics`` acceptance suite.

Pure builders and runner fakes live in ``metrics.helpers``; this module only
wires up pytest fixtures (auto-discovered by directory). The planned
``dr_code.metrics`` package does not exist yet, so this file imports only from
packages that do (``dr_code.trace``, ``dr_code.humaneval.*``, ``metrics.helpers``).
Every symbol from ``dr_code.metrics`` is imported lazily inside the test that
exercises it, so the suite collects cleanly against the missing package and
fails hard — never skips — when it is absent.

Nothing here touches a real container runtime.
"""

from __future__ import annotations

import pytest

from dr_code.humaneval.sandbox import SandboxRunner
from dr_code.humaneval.task import HumanEvalTask

from metrics.helpers import (
    CountingRunner,
    local_runner as _local_runner,
    make_task,
)


@pytest.fixture
def task() -> HumanEvalTask:
    return make_task()


@pytest.fixture
def good_submission() -> str:
    """A submission that passes every case of :func:`make_task`."""
    return "def add_one(x):\n    return x + 1\n"


@pytest.fixture
def failing_submission() -> str:
    """A submission that compiles and runs but fails assertions."""
    return "def add_one(x):\n    return x - 1\n"


@pytest.fixture
def local_runner() -> SandboxRunner:
    """Injectable host-interpreter runner (no OCI container)."""
    return _local_runner()


@pytest.fixture
def counting_runner(local_runner: SandboxRunner) -> CountingRunner:
    """A counting wrapper around the local runner (observes at-most-once)."""
    return CountingRunner(local_runner)
