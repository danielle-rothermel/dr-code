"""Testing-stage configuration."""

from __future__ import annotations

import os
from typing import Final

from nl_code.code_execution.models import DEFAULT_CODE_EVAL_IMAGE

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0


def default_timeout_seconds() -> float:
    raw = os.environ.get("DR_CODE_TEST_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    return float(raw)


def default_docker_image() -> str | None:
    return os.environ.get("DR_CODE_DOCKER_IMAGE") or DEFAULT_CODE_EVAL_IMAGE
