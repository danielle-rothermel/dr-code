from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dr_exec import FakeExecutor

_HERE = str(Path(__file__).parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _executor_stubs import local_python_executor  # noqa: E402


@pytest.fixture
def local_executor() -> FakeExecutor:
    return local_python_executor()
