from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from dr_code.core.execution.sandbox import (
    SandboxOutputLimitError,
    SandboxRunner,
    SandboxTimeoutError,
)
from dr_code.core.models import FrozenModel

_TIMEOUT_RETURNCODE = -100_000_001
_OUTPUT_LIMIT_RETURNCODE = -100_000_002


class ExecutionRequest(FrozenModel):
    """One cache-keyed sandbox request."""

    source: str
    input_json: str
    timeout_seconds: float
    computation_id: str


class ExecutionOutcome(FrozenModel):
    returncode: int
    stdout: str
    stderr: str


class ExecutionCache(Protocol):
    """Cache keys omit the injected runner; scope caches to one runtime."""

    def get(self, key: str) -> ExecutionOutcome | None: ...

    def put(self, key: str, outcome: ExecutionOutcome) -> None: ...


class InMemoryExecutionCache:
    def __init__(self) -> None:
        self._outcomes: dict[str, ExecutionOutcome] = {}

    def get(self, key: str) -> ExecutionOutcome | None:
        return self._outcomes.get(key)

    def put(self, key: str, outcome: ExecutionOutcome) -> None:
        self._outcomes[key] = outcome


def run_requests(
    requests: Sequence[ExecutionRequest],
    *,
    run_in_sandbox: SandboxRunner,
    cache: ExecutionCache,
) -> dict[ExecutionRequest, ExecutionOutcome]:
    """Deduplicate requests within this call and reuse cache hits."""

    outcomes: dict[ExecutionRequest, ExecutionOutcome] = {}
    unique_requests: dict[str, ExecutionRequest] = {}
    for request in requests:
        unique_requests.setdefault(_request_cache_key(request), request)

    for key, request in unique_requests.items():
        cached = cache.get(key)
        if cached is not None:
            outcomes[request] = cached
            continue
        try:
            completed = run_in_sandbox(
                source=request.source,
                input_json=request.input_json,
                timeout_seconds=request.timeout_seconds,
            )
            outcome = ExecutionOutcome(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except SandboxTimeoutError as exc:
            outcome = ExecutionOutcome(
                returncode=_TIMEOUT_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        except SandboxOutputLimitError as exc:
            outcome = ExecutionOutcome(
                returncode=_OUTPUT_LIMIT_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        cache.put(key, outcome)
        outcomes[request] = outcome
    return outcomes


def _request_cache_key(request: ExecutionRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_timeout_outcome(outcome: ExecutionOutcome) -> bool:
    return outcome.returncode == _TIMEOUT_RETURNCODE


def is_output_limit_outcome(outcome: ExecutionOutcome) -> bool:
    return outcome.returncode == _OUTPUT_LIMIT_RETURNCODE
