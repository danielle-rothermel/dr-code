"""Execution planning, caching, and request-deduplication contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.core.execution.sandbox import (
    SandboxCompletedProcess,
    SandboxError,
    SandboxOutputLimitError,
    SandboxTimeoutError,
)
from dr_code.trace import CodeArtifact, TextArtifact, external_trace

from ._helpers import _definition, _extract, _extract_batch, _facts, _q


def _request(
    *,
    source: str = "print('hi')",
    input_json: str = "{}",
    timeout_seconds: float = 1.0,
    computation_id: str = "humaneval-runner@0",
):
    from dr_code.metrics.engine.execution import ExecutionRequest

    return ExecutionRequest(
        source=source,
        input_json=input_json,
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

    def __call__(self, *, source, input_json, timeout_seconds):  # noqa: ANN001
        self.calls += 1
        return SandboxCompletedProcess(
            returncode=0, stdout=self._stdout, stderr=""
        )


def _run(requests, *, runner=None, cache=None):
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        run_requests,
    )

    return run_requests(
        requests,
        run_in_sandbox=runner or _CountingRunner(),
        cache=cache or InMemoryExecutionCache(),
    )


def test_execution_outcome_holds_sandbox_completed_process_fields() -> None:
    outcome = _outcome(returncode=0, stdout="[{}]", stderr="warn")
    assert outcome.returncode == 0
    assert outcome.stdout == "[{}]"
    assert outcome.stderr == "warn"


def test_execution_outcome_is_frozen() -> None:
    outcome = _outcome()
    with pytest.raises(ValidationError) as exc_info:
        outcome.returncode = 1  # type: ignore[misc]

    assert [
        (error["type"], error["loc"]) for error in exc_info.value.errors()
    ] == [("frozen_instance", ("returncode",))]


def test_execution_outcome_is_json_serializable() -> None:
    import json

    parsed = json.loads(
        _outcome(stdout='[{"case_id": "0"}]').model_dump_json()
    )
    assert parsed["returncode"] == 0


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
        ("input_json", "{}", '{"x":1}'),
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

    def raising(*, source, input_json, timeout_seconds):  # noqa: ANN001
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
    """SandboxError infrastructure failures propagate through run_requests."""

    def infra(*, source, input_json, timeout_seconds):  # noqa: ANN001
        raise SandboxError("infra failed")

    with pytest.raises(SandboxError):
        _run([_request()], runner=infra)


def test_run_requests_caches_output_limit_outcome() -> None:
    """Candidate output flooding becomes a reusable sentinel outcome."""
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        is_output_limit_outcome,
    )

    calls = 0

    def output_limited(*, source, input_json, timeout_seconds):  # noqa: ANN001
        nonlocal calls
        calls += 1
        raise SandboxOutputLimitError("output limit reached")

    request = _request()
    cache = InMemoryExecutionCache()

    first = _run([request], runner=output_limited, cache=cache)
    second = _run([request], runner=output_limited, cache=cache)

    assert calls == 1
    assert second == first
    assert is_output_limit_outcome(second[request])
    assert second[request].stdout == ""


def test_planning_sandbox_error_propagates_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    counting_runner,
) -> None:
    """Infrastructure failure during planning aborts before sandbox work."""
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    operator_cls = REGISTRY[str(MetricName.TEXT_STATS)]

    def fail_planning(self, value, aux):  # noqa: ANN001
        raise SandboxError("planning infrastructure failed")

    monkeypatch.setattr(operator_cls, "execution_requests", fail_planning)

    with pytest.raises(SandboxError):
        _extract(
            _definition([_q("text_stats")]),
            trace,
            run_in_sandbox=counting_runner,
        )
    assert counting_runner.call_count == 0


def test_infrastructure_sandbox_error_raises(
    task, code_test_trace, raising_runner
) -> None:
    """A SandboxError is infra breakage — it raises, never becomes a record."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(SandboxError):
        _extract(
            definition,
            trace,
            run_in_sandbox=raising_runner(SandboxError("infra broke")),
        )


def test_missing_execution_outcome_raises_engine_invariant_error(
    task,
    local_runner,
    code_test_trace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If CodeTest.compute rebuilds an ExecutionRequest that diverges from
    what execution_requests planned, ctx.outcome_for's lookup misses. That
    is an engine bug, not a metric bug: it must surface as
    EngineInvariantError out of the batch, never get swallowed into an
    operator_failure record."""
    from dr_code.metrics import EngineInvariantError
    from dr_code.humaneval.metric_operator import CodeTest

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    def no_requests(self, value, aux):  # noqa: ANN001
        return ()

    monkeypatch.setattr(CodeTest, "execution_requests", no_requests)
    with pytest.raises(EngineInvariantError):
        _extract(definition, trace, run_in_sandbox=local_runner)


def test_sandbox_timeout_is_candidate_data_not_infrastructure(
    task, code_test_trace, raising_runner
) -> None:
    """A candidate timeout is attributed to the candidate as data (timeout
    cases), not a raised SandboxError (runner attribution parity)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=1.0)]
    )
    record = _extract(
        definition,
        trace,
        run_in_sandbox=raising_runner(SandboxTimeoutError("timed out")),
    )[0]
    assert record.status.value == "measured"
    facts = _facts(record)
    assert facts["timeout_count"] == facts["total_cases"]


def test_batch_dedupes_identical_code_test_executions(
    task, counting_runner, code_test_trace
) -> None:
    """Identical submissions across a sweep execute once (at-most-once)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(
        definition, [trace, trace, trace], run_in_sandbox=counting_runner
    )
    assert counting_runner.call_count == 1


def test_distinct_submissions_execute_separately(
    task, counting_runner, code_test_trace
) -> None:
    good = code_test_trace("def add_one(x):\n    return x + 1\n", task)
    bad = code_test_trace("def add_one(x):\n    return x - 1\n", task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(definition, [good, bad], run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 2


def test_batch_returns_one_record_tuple_per_trace(
    task, local_runner, code_test_trace
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=5.0)]
    )
    results = _extract_batch(
        definition, [trace, trace], run_in_sandbox=local_runner
    )
    assert isinstance(results, tuple)
    assert len(results) == 2
    for per_trace in results:
        assert isinstance(per_trace, tuple)
        assert len(per_trace) == 1


def test_prepopulated_execution_cache_skips_the_runner(
    task, counting_runner, code_test_trace
) -> None:
    """A cache hit means the injected runner is never called."""
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    cache = InMemoryExecutionCache()
    _extract(
        definition,
        trace,
        run_in_sandbox=counting_runner,
        execution_cache=cache,
    )
    assert counting_runner.call_count == 1

    counting_runner.calls.clear()
    _extract(
        definition,
        trace,
        run_in_sandbox=counting_runner,
        execution_cache=cache,
    )
    assert counting_runner.call_count == 0


def test_pure_operators_never_call_the_runner(counting_runner) -> None:
    """Pure operators declare no execution requests."""
    text = "def f(x):\n    return x\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=text),
            "output": CodeArtifact(source=text),
        }
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )
    _extract_batch(definition, [trace, trace], run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_code_test_record_values_exclude_timing(
    task, local_runner, code_test_trace
) -> None:
    """Determinism soft spot: timing stays out of record values so identical
    inputs reproduce."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=2.0)]
    )
    record = _extract(definition, trace, run_in_sandbox=local_runner)[0]
    assert "elapsed_seconds" not in _facts(record)
