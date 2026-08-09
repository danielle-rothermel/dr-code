from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from dr_exec import Executor

from dr_code.core.execution.executor import (
    ExecutionKilledError,
    ExecutionOutputLimitError,
    ExecutionTimeoutError,
    run_python_source,
)
from dr_code.core.models import FrozenModel

_TIMEOUT_RETURNCODE = -100_000_001
_OUTPUT_LIMIT_RETURNCODE = -100_000_002
_KILLED_RETURNCODE = -100_000_003
_EXECUTION_REQUEST_CACHE_KEY_VERSION = 1


class ExecutionRequest(FrozenModel):
    """One cache-keyed execution request."""

    source: str
    input_json: str
    timeout_seconds: float
    computation_id: str


class ExecutionOutcome(FrozenModel):
    returncode: int
    stdout: str
    stderr: str


class ExecutionCache(Protocol):
    """Cache opaque request keys; persistent adapters own scope identity."""

    async def prefetch(self, keys: Sequence[str]) -> None: ...

    def get(self, key: str) -> ExecutionOutcome | None: ...

    async def put(self, key: str, outcome: ExecutionOutcome) -> None: ...


class InMemoryExecutionCache:
    def __init__(self) -> None:
        self._outcomes: dict[str, ExecutionOutcome] = {}

    async def prefetch(self, keys: Sequence[str]) -> None:
        pass

    def get(self, key: str) -> ExecutionOutcome | None:
        return self._outcomes.get(key)

    async def put(self, key: str, outcome: ExecutionOutcome) -> None:
        self._outcomes[key] = outcome


async def run_requests(
    requests: Sequence[ExecutionRequest],
    *,
    executor: Executor | None,
    cache: ExecutionCache,
) -> dict[ExecutionRequest, ExecutionOutcome]:
    """Deduplicate requests within this call and reuse cache hits."""

    outcomes: dict[ExecutionRequest, ExecutionOutcome] = {}
    unique_requests: dict[str, ExecutionRequest] = {}
    text_digests: dict[str, str] = {}
    for request in requests:
        key = _execution_request_cache_key(request, text_digests)
        unique_requests.setdefault(key, request)

    await cache.prefetch(tuple(unique_requests))
    for key, request in unique_requests.items():
        cached = cache.get(key)
        if cached is not None:
            outcomes[request] = cached
            continue
        try:
            completed = run_python_source(
                executor,
                source=request.source,
                input_json=request.input_json,
                timeout_seconds=request.timeout_seconds,
            )
            outcome = ExecutionOutcome(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except ExecutionTimeoutError as exc:
            outcome = ExecutionOutcome(
                returncode=_TIMEOUT_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        except ExecutionOutputLimitError as exc:
            outcome = ExecutionOutcome(
                returncode=_OUTPUT_LIMIT_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        except ExecutionKilledError as exc:
            outcome = ExecutionOutcome(
                returncode=_KILLED_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        await cache.put(key, outcome)
        outcomes[request] = outcome
    return outcomes


def execution_request_cache_key(request: ExecutionRequest) -> str:
    """Return the versioned request key, excluding persistent cache scope."""

    return _execution_request_cache_key(request, {})


def _execution_request_cache_key(
    request: ExecutionRequest,
    text_digests: dict[str, str],
) -> str:
    payload = json.dumps(
        {
            "version": _EXECUTION_REQUEST_CACHE_KEY_VERSION,
            "source_sha256": _text_digest(request.source, text_digests),
            "input_json_sha256": _text_digest(
                request.input_json, text_digests
            ),
            "timeout_seconds": request.timeout_seconds,
            "computation_id": request.computation_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_digest(value: str, text_digests: dict[str, str]) -> str:
    digest = text_digests.get(value)
    if digest is None:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        text_digests[value] = digest
    return digest


def is_timeout_outcome(outcome: ExecutionOutcome) -> bool:
    return outcome.returncode == _TIMEOUT_RETURNCODE


def is_output_limit_outcome(outcome: ExecutionOutcome) -> bool:
    return outcome.returncode == _OUTPUT_LIMIT_RETURNCODE


def is_killed_outcome(outcome: ExecutionOutcome) -> bool:
    return outcome.returncode == _KILLED_RETURNCODE
