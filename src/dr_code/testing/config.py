"""Testing-stage configuration."""

from __future__ import annotations

import os
from typing import Final

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
TEST_TIMEOUT_SECONDS_ENV: Final[str] = "DR_CODE_TEST_TIMEOUT_SECONDS"


def default_timeout_seconds() -> float:
    raw = os.environ.get(TEST_TIMEOUT_SECONDS_ENV)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    return float(raw)
