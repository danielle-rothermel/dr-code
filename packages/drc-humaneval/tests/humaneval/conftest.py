from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dr_exec import FakeExecutor

_HUMANEVAL_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _HUMANEVAL_TESTS_ROOT not in sys.path:
    sys.path.insert(0, _HUMANEVAL_TESTS_ROOT)

from _executor_stubs import local_python_executor  # noqa: E402


@pytest.fixture
def local_executor() -> FakeExecutor:
    return local_python_executor()
