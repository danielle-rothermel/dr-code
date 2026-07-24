"""Content-addressed execution requests and cache orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Protocol

from dr_code.execution.subprocess import (
    PythonSubprocessRunner,
    SubprocessOutputLimitError,
    SubprocessTimeoutError,
)
from dr_code.models import FrozenModel
from dr_code.trace import stable_hash

_TIMEOUT_RETURNCODE = -100_000_001
_OUTPUT_LIMIT_RETURNCODE = -100_000_002


class ExecutionRequest(FrozenModel):
    """One deterministic invocation of a trusted subprocess runner program."""

    source: str
    input_text: str
    timeout_seconds: float
    computation_id: str

    @cached_property
    def cache_key(self) -> str:
        """Return a content hash over every execution-affecting field."""

        return stable_hash(self)


class ExecutionOutcome(FrozenModel):
    """The cacheable fields returned by the subprocess execution boundary."""

    returncode: int
    stdout: str
    stderr: str


class ExecutionCache(Protocol):
    """Outcome cache keyed by ``ExecutionRequest.cache_key``.

    The key hashes only the request fields; the injected
    ``PythonSubprocessRunner`` is not part of it. A cache instance must
    therefore be scoped to a single runner/runtime -- sharing one across
    runners can silently return an outcome produced by a different execution
    environment.
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
    run_in_subprocess: PythonSubprocessRunner,
    cache: ExecutionCache,
) -> dict[str, ExecutionOutcome]:
    """Execute each distinct cache miss at most once.

    Timeouts and output-limit failures are candidate-attributable outcomes.
    Other subprocess failures remain infrastructure exceptions and propagate.
    """

    outcomes: dict[str, ExecutionOutcome] = {}
    unique_requests: dict[str, ExecutionRequest] = {}
    for request in requests:
        unique_requests.setdefault(request.cache_key, request)

    for key, request in unique_requests.items():
        cached = cache.get(key)
        if cached is not None:
            outcomes[key] = cached
            continue
        try:
            completed = run_in_subprocess(
                source=request.source,
                input_text=request.input_text,
                timeout_seconds=request.timeout_seconds,
            )
            outcome = ExecutionOutcome(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except SubprocessTimeoutError as exc:
            outcome = ExecutionOutcome(
                returncode=_TIMEOUT_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        except SubprocessOutputLimitError as exc:
            outcome = ExecutionOutcome(
                returncode=_OUTPUT_LIMIT_RETURNCODE,
                stdout="",
                stderr=str(exc),
            )
        cache.put(key, outcome)
        outcomes[key] = outcome
    return outcomes


def is_timeout_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether an outcome represents a candidate wall-clock timeout."""

    return outcome.returncode == _TIMEOUT_RETURNCODE


def is_output_limit_outcome(outcome: ExecutionOutcome) -> bool:
    """Whether an outcome represents candidate output flooding."""

    return outcome.returncode == _OUTPUT_LIMIT_RETURNCODE
