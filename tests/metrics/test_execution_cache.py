"""Execution-cache contracts.

Covers ``ExecutionRequest`` content-hash ``cache_key`` (deterministic,
content-addressed), ``ExecutionOutcome`` (``SubprocessCompletedProcess``
fields),
the ``ExecutionCache`` protocol plus ``InMemoryExecutionCache`` get/put, and
``run_requests`` deduplication with at-most-once execution per cache lifetime.

A counting fake runner stands in for the injected ``PythonSubprocessRunner``.
"""

from __future__ import annotations

import pytest

from dr_code.execution.subprocess import (
    SubprocessCompletedProcess,
    SubprocessError,
    SubprocessOutputLimitError,
    SubprocessTimeoutError,
)


def _request(
    *,
    source: str = "print('hi')",
    input_text: str = "{}",
    timeout_seconds: float = 1.0,
    computation_id: str = "humaneval-runner@v1",
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
# ExecutionRequest.cache_key — content-addressed and deterministic.
# ===========================================================================


def test_cache_key_is_a_deterministic_string() -> None:
    assert isinstance(_request().cache_key, str)
    assert len(_request().cache_key) > 0
    assert _request().cache_key == _request().cache_key


@pytest.mark.parametrize(
    ("field", "a", "b"),
    [
        ("source", "a", "b"),
        ("input_text", "{}", '{"x":1}'),
        ("timeout_seconds", 1.0, 2.0),
        ("computation_id", "a@v1", "a@v2"),
    ],
)
def test_cache_key_depends_on_each_request_field(field, a, b) -> None:
    assert _request(**{field: a}).cache_key != _request(**{field: b}).cache_key


# ===========================================================================
# ExecutionOutcome — SubprocessCompletedProcess fields, frozen.
# ===========================================================================


def test_execution_outcome_holds_subprocess_completed_process_fields() -> None:
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

    request = _request()
    outcome = _outcome()
    cache = InMemoryExecutionCache()
    assert cache.get(request.cache_key) is None
    cache.put(request.cache_key, outcome)
    assert cache.get(request.cache_key) == outcome


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
    assert set(outcomes) == {requests[0].cache_key}


def test_run_requests_executes_distinct_requests() -> None:
    runner = _CountingRunner()
    _run([_request(input_text="a"), _request(input_text="b")], runner=runner)
    assert runner.calls == 2


def test_run_requests_serves_cache_hits_without_running() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    request = _request()
    cache = InMemoryExecutionCache()
    outcome = _outcome()
    cache.put(request.cache_key, outcome)  # pre-populate

    runner = _CountingRunner()
    outcomes = _run([request], runner=runner, cache=cache)
    assert runner.calls == 0
    assert outcomes[request.cache_key] == outcome


def test_run_requests_populates_cache_for_misses() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    cache = InMemoryExecutionCache()
    request = _request()
    _run([request], cache=cache)
    assert cache.get(request.cache_key) is not None


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
    assert cache.get(request.cache_key) is None


def test_run_requests_subprocess_error_propagates() -> None:
    """SubprocessError infrastructure failures propagate through run_requests."""

    def infra(*, source, input_text, timeout_seconds):  # noqa: ANN001
        raise SubprocessError("infra failed")

    with pytest.raises(SubprocessError):
        _run([_request()], runner=infra)


def test_run_requests_timeout_becomes_cacheable_candidate_outcome() -> None:
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        is_timeout_outcome,
    )

    def timed_out(*, source, input_text, timeout_seconds):  # noqa: ANN001
        raise SubprocessTimeoutError("timed out")

    cache = InMemoryExecutionCache()
    request = _request()
    outcomes = _run([request], runner=timed_out, cache=cache)
    outcome = outcomes[request.cache_key]
    assert is_timeout_outcome(outcome)
    assert cache.get(request.cache_key) == outcome


def test_run_requests_output_limit_becomes_cacheable_candidate_outcome() -> (
    None
):
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        is_output_limit_outcome,
    )

    def flooded(*, source, input_text, timeout_seconds):  # noqa: ANN001
        raise SubprocessOutputLimitError("too much output")

    cache = InMemoryExecutionCache()
    request = _request()
    outcomes = _run([request], runner=flooded, cache=cache)
    outcome = outcomes[request.cache_key]
    assert is_output_limit_outcome(outcome)
    assert cache.get(request.cache_key) == outcome
