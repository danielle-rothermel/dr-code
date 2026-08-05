"""Execution requests and private cache orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol

from dr_code.humaneval.sandbox import (
    CANDIDATE_KILL_RETURNCODES,
    SandboxOutputLimitError,
    SandboxRunner,
    SandboxTimeoutError,
)
from dr_code.base import FrozenModel

_TIMEOUT_RETURNCODE = -100_000_001
_OUTPUT_LIMIT_RETURNCODE = -100_000_002


class ExecutionRequest(FrozenModel):
    """One deterministic invocation of a trusted sandbox runner program."""

    source: str
    input_json: str
    timeout_seconds: float
    computation_id: str


class ExecutionOutcome(FrozenModel):
    """The cacheable fields returned by the sandbox execution boundary."""

    returncode: int
    stdout: str
    stderr: str


class ExecutionCache(Protocol):
    """Outcome cache keyed by an opaque request implementation detail.

    The key hashes only the request fields; the injected ``SandboxRunner``
    is not part of it. A cache instance must therefore be scoped to a single
    runner/runtime -- sharing one across runners can silently return an
    outcome produced by a different execution environment.
    """

    def get(self, key: str) -> ExecutionOutcome | None: ...

    def put(self, key: str, outcome: ExecutionOutcome) -> None: ...


class InMemoryExecutionCache:
    """A process-local execution outcome cache."""

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
    """Execute each distinct cache miss at most once.

    Timeouts and output-limit failures are candidate-attributable outcomes.
    Other sandbox failures remain infrastructure exceptions and propagate.
    """

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
    """Hash a canonical request only for this module's cache lookup."""

    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_timeout_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether an outcome represents a candidate wall-clock timeout."""

    return outcome.returncode == _TIMEOUT_RETURNCODE


def is_output_limit_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether an outcome represents candidate output flooding."""

    return outcome.returncode == _OUTPUT_LIMIT_RETURNCODE


def is_candidate_kill_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether an outcome represents a candidate-provoked hard kill.

    A sibling of the timeout / output-limit predicates: the kill returncodes
    (e.g. SIGKILL/SIGSEGV) are things candidate code can provoke, so this is
    candidate-attributable data, not infrastructure breakage.
    """

    return outcome.returncode in CANDIDATE_KILL_RETURNCODES
