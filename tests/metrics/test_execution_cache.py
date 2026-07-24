"""Execution-cache contracts.

Covers ``ExecutionOutcome`` boundary fields, the ``ExecutionCache`` protocol,
and observable request deduplication with at-most-once execution per cache
lifetime. Cache keys remain a private implementation detail.

A counting fake runner stands in for the injected ``PythonSubprocessRunner``.
"""

from __future__ import annotations

import pytest

from dr_code.execution.subprocess import (
    SubprocessCompletedProcess,
    SubprocessError,
)


def _request(
    *,
    source: str = "print('hi')",
    input_text: str = "{}",
    timeout_seconds: float = 1.0,
    computation_id: str = "humaneval-runner@0",
):
    from dr_code.metrics.engine.execution import ExecutionRequest

    return ExecutionRequest(
        source=source,
        input_text=input_text,
        timeout_seconds=timeout_seconds,
        computation_id=computation_id,
    )


def _outcome(
    *,
    returncode: int = 0,
    stdout: str = "[]",
    stderr: str = "",
):
    from dr_code.metrics.engine.execution import ExecutionOutcome

    return ExecutionOutcome(
        returncode=returncode, stdout=stdout, stderr=stderr
    )


class _CountingRunner:
    def __init__(self, stdout: str = "[]") -> None:
        self.calls = 0
        self._stdout = stdout

    def __call__(self, *, source, input_text, timeout_seconds):  # noqa: ANN001
        self.calls += 1
        return SubprocessCompletedProcess(
            returncode=0, stdout=self._stdout, stderr=""
        )


def _run(requests, *, runner=None, cache=None):
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        run_requests,
    )

    return run_requests(
        requests,
        run_in_subprocess=runner or _CountingRunner(),
        cache=cache or InMemoryExecutionCache(),
    )


# ===========================================================================
# ExecutionOutcome — SubprocessCompletedProcess fields, frozen.
# ===========================================================================


def test_execution_outcome_holds_sandbox_completed_process_fields() -> None:
    outcome = _outcome(returncode=0, stdout="[{}]", stderr="warn")
    assert outcome.returncode == 0
    assert outcome.stdout == "[{}]"
    assert outcome.stderr == "warn"


def test_execution_outcome_is_frozen() -> None:
    outcome = _outcome()
    with pytest.raises(Exception):  # noqa: PT011
        outcome.returncode = 1  # type: ignore[misc]


def test_execution_outcome_is_json_serializable() -> None:
    import json

    parsed = json.loads(
        _outcome(stdout='[{"case_id": "0"}]').model_dump_json()
    )
    assert parsed["returncode"] == 0


# ===========================================================================
# InMemoryExecutionCache get/put.
# ===========================================================================


def test_in_memory_cache_miss_returns_none() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    assert InMemoryExecutionCache().get("nope") is None


def test_in_memory_cache_get_put_round_trip() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    outcome = _outcome()
    cache = InMemoryExecutionCache()
    assert cache.get("opaque-key") is None
    cache.put("opaque-key", outcome)
    assert cache.get("opaque-key") == outcome


def test_in_memory_cache_put_overwrites() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    cache = InMemoryExecutionCache()
    first = _outcome(returncode=0, stdout="a")
    second = _outcome(returncode=1, stdout="b")
    cache.put("k", first)
    cache.put("k", second)
    assert cache.get("k") == second


# ===========================================================================
# run_requests — dedupe + at-most-once execution.
# ===========================================================================


def test_run_requests_dedupes_identical_requests() -> None:
    runner = _CountingRunner()
    requests = [_request(), _request(), _request()]
    outcomes = _run(requests, runner=runner)
    assert runner.calls == 1
    # every duplicate resolves to the single executed outcome
    assert len(outcomes) == 1
    assert set(outcomes) == {requests[0]}


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("source", "a", "b"),
        ("input_text", "{}", '{"x":1}'),
        ("timeout_seconds", 1.0, 2.0),
        ("computation_id", "runner-a", "runner-b"),
    ],
)
def test_run_requests_does_not_alias_distinct_requests(
    field: str,
    first: object,
    second: object,
) -> None:
    runner = _CountingRunner()
    requests = [
        _request(**{field: first}),
        _request(**{field: second}),
    ]

    outcomes = _run(requests, runner=runner)

    assert runner.calls == 2
    assert set(outcomes) == set(requests)


def test_run_requests_serves_cache_hits_without_reexecuting() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    request = _request()
    cache = InMemoryExecutionCache()
    runner = _CountingRunner()
    first = _run([request], runner=runner, cache=cache)
    second = _run([request], runner=runner, cache=cache)

    assert runner.calls == 1
    assert second == first
    assert second[request] == _outcome()


def test_run_requests_empty_input_returns_empty_dict() -> None:
    assert _run([]) == {}


def test_run_requests_runner_error_propagates_and_is_not_cached() -> None:
    """A runner exception propagates; the failing outcome is never stored."""
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    def raising(*, source, input_text, timeout_seconds):  # noqa: ANN001
        raise RuntimeError("runner exploded")

    cache = InMemoryExecutionCache()
    request = _request()
    with pytest.raises(RuntimeError):
        _run([request], runner=raising, cache=cache)

    runner = _CountingRunner()
    outcomes = _run([request], runner=runner, cache=cache)
    assert runner.calls == 1
    assert outcomes[request] == _outcome()


def test_run_requests_sandbox_error_propagates() -> None:
    """SubprocessError infrastructure failures propagate through run_requests."""

    def infra(*, source, input_text, timeout_seconds):  # noqa: ANN001
        raise SubprocessError("infra failed")

    with pytest.raises(SubprocessError):
        _run([_request()], runner=infra)
