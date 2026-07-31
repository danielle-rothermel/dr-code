"""Pytest fixtures for the ``dr_code.metrics`` acceptance suite.

Pure builders and executor doubles live in ``metrics.helpers``; this module
only wires up pytest fixtures (auto-discovered by directory). Every symbol from
``dr_code.metrics`` is imported lazily inside the test that exercises it, so the
suite collects cleanly and fails hard — never skips — when a package is absent.

Logic tests drive a scripted ``FakeExecutor``; parity and oracle tests drive the
real batch executor with ``Records.none()``.
"""

from __future__ import annotations

import pytest

from dr_code.humaneval.batch_runner import PRODUCTION_EXECUTOR
from dr_code.humaneval.task import HumanEvalTask

from metrics.helpers import CountingExecutor, make_task


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
def real_executor() -> object:
    """The real dr-exec batch executor (records disabled per call site)."""
    return PRODUCTION_EXECUTOR


@pytest.fixture
def counting_executor() -> CountingExecutor:
    """A counting wrapper around the real executor (observes at-most-once)."""
    return CountingExecutor(PRODUCTION_EXECUTOR)
