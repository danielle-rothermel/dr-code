from __future__ import annotations

import pytest

from _executor_stubs import CountingExecutor, local_python_executor
from metrics.operators._helpers import _text_trace


@pytest.fixture
def counting_executor() -> CountingExecutor:
    return CountingExecutor(local_python_executor())


@pytest.fixture
def text_trace():
    return _text_trace
