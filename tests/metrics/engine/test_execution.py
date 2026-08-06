from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from _executor_stubs import (
    CountingExecutor,
    output_limit_executor,
    raising_executor,
    scripted_executor,
    timeout_executor,
)
from dr_exec import ExecutorFailure
from dr_code.trace import CodeArtifact, TextArtifact, external_trace

from ._helpers import _definition, _extract, _extract_batch, _facts, _q


def _request(
    *,
    source: str = "def dr_exec_main(request, emit):\n    pass\n",
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


def _counting_stub(stdout: str = "[]") -> CountingExecutor:
    return CountingExecutor(scripted_executor(stdout=stdout))


def _run(requests, *, executor=None, cache=None):
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        run_requests,
    )

    return run_requests(
        requests,
        executor=executor if executor is not None else _counting_stub(),
        cache=cache or InMemoryExecutionCache(),
    )


def test_execution_outcome_holds_completed_process_fields() -> None:
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


def test_in_memory_cache_prefetch_is_a_no_op() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    cache = InMemoryExecutionCache()
    outcome = _outcome()
    cache.put("present", outcome)

    cache.prefetch(("present", "missing"))

    assert cache.get("present") == outcome
    assert cache.get("missing") is None


def test_run_requests_dedupes_identical_requests() -> None:
    executor = _counting_stub()
    requests = [_request(), _request(), _request()]
    outcomes = _run(requests, executor=executor)
    assert executor.call_count == 1

    assert len(outcomes) == 1
    assert set(outcomes) == {requests[0]}


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        (
            "source",
            "def dr_exec_main(request, emit):\n    pass\n",
            "def dr_exec_main(request, emit):\n    return\n",
        ),
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
    from dr_code.metrics.engine.execution import execution_request_cache_key

    executor = _counting_stub()
    requests = [
        _request(**{field: first}),
        _request(**{field: second}),
    ]

    assert execution_request_cache_key(
        requests[0]
    ) != execution_request_cache_key(requests[1])

    outcomes = _run(requests, executor=executor)

    assert executor.call_count == 2
    assert set(outcomes) == set(requests)


def test_execution_request_cache_key_uses_compact_versioned_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from dr_code.metrics.engine import execution

    source = "source-marker-" * 100
    input_json = "input-marker-" * 100
    request = _request(source=source, input_json=input_json)
    hashed_values: list[bytes] = []
    real_sha256 = execution.hashlib.sha256

    def recording_sha256(value: bytes = b""):
        hashed_values.append(value)
        return real_sha256(value)

    monkeypatch.setattr(execution.hashlib, "sha256", recording_sha256)

    key = execution.execution_request_cache_key(request)

    expected_payload = json.dumps(
        {
            "version": 1,
            "source_sha256": real_sha256(source.encode()).hexdigest(),
            "input_json_sha256": real_sha256(input_json.encode()).hexdigest(),
            "timeout_seconds": request.timeout_seconds,
            "computation_id": request.computation_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashed_values == [
        source.encode(),
        input_json.encode(),
        expected_payload,
    ]
    assert key == real_sha256(expected_payload).hexdigest()


def test_run_requests_prefetches_deduplicated_keys_before_gets() -> None:
    from dr_code.metrics.engine.execution import (
        ExecutionOutcome,
        execution_request_cache_key,
    )

    class RecordingCache:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def prefetch(self, keys: Sequence[str]) -> None:
            self.events.append(("prefetch", tuple(keys)))

        def get(self, key: str) -> ExecutionOutcome | None:
            self.events.append(("get", key))
            return None

        def put(self, key: str, outcome: ExecutionOutcome) -> None:
            self.events.append(("put", key))

    first = _request()
    second = _request(computation_id="second-runner")
    cache = RecordingCache()

    _run([first, first, second], cache=cache)

    expected_keys = tuple(
        execution_request_cache_key(request) for request in (first, second)
    )
    assert cache.events[0] == ("prefetch", expected_keys)
    assert [event for event in cache.events if event[0] == "get"] == [
        ("get", key) for key in expected_keys
    ]
    assert sum(event[0] == "prefetch" for event in cache.events) == 1


def test_run_requests_memoizes_repeated_text_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics.engine import execution

    source = "def dr_exec_main(request, emit):\n" + "    pass\n" * 1_000
    input_json = '{"items":[' + ",".join("0" for _ in range(1_000)) + "]}"
    hashed_values: list[bytes] = []
    real_sha256 = execution.hashlib.sha256

    def recording_sha256(value: bytes = b""):
        hashed_values.append(value)
        return real_sha256(value)

    monkeypatch.setattr(execution.hashlib, "sha256", recording_sha256)

    _run(
        [
            _request(source=source, input_json=input_json),
            _request(
                source=source,
                input_json=input_json,
                computation_id="second-runner",
            ),
        ]
    )

    assert hashed_values.count(source.encode("utf-8")) == 1
    assert hashed_values.count(input_json.encode("utf-8")) == 1


def test_run_requests_serves_cache_hits_without_reexecuting() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    request = _request()
    cache = InMemoryExecutionCache()
    executor = _counting_stub()
    first = _run([request], executor=executor, cache=cache)
    second = _run([request], executor=executor, cache=cache)

    assert executor.call_count == 1
    assert second == first
    assert second[request] == _outcome()


def test_run_requests_cache_hits_need_no_executor() -> None:
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        run_requests,
    )

    request = _request()
    cache = InMemoryExecutionCache()
    _run([request], cache=cache)

    outcomes = run_requests([request], executor=None, cache=cache)
    assert outcomes[request] == _outcome()


def test_run_requests_without_executor_fails_closed() -> None:
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        run_requests,
    )

    with pytest.raises(ExecutorFailure, match="no executor"):
        run_requests(
            [_request()],
            executor=None,
            cache=InMemoryExecutionCache(),
        )


def test_run_requests_empty_input_returns_empty_dict() -> None:
    assert _run([]) == {}


def test_run_requests_runner_error_propagates_and_is_not_cached() -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    cache = InMemoryExecutionCache()
    request = _request()
    with pytest.raises(RuntimeError):
        _run(
            [request],
            executor=raising_executor(RuntimeError("runner exploded")),
            cache=cache,
        )

    executor = _counting_stub()
    outcomes = _run([request], executor=executor, cache=cache)
    assert executor.call_count == 1
    assert outcomes[request] == _outcome()


def test_run_requests_executor_failure_propagates() -> None:
    with pytest.raises(ExecutorFailure):
        _run(
            [_request()],
            executor=raising_executor(ExecutorFailure("infra failed")),
        )


def test_run_requests_caches_timeout_outcome() -> None:
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        is_timeout_outcome,
    )

    request = _request()
    cache = InMemoryExecutionCache()
    executor = CountingExecutor(timeout_executor())

    first = _run([request], executor=executor, cache=cache)
    second = _run([request], executor=executor, cache=cache)

    assert executor.call_count == 1
    assert second == first
    assert is_timeout_outcome(second[request])


def test_run_requests_caches_output_limit_outcome() -> None:
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        is_output_limit_outcome,
    )

    request = _request()
    cache = InMemoryExecutionCache()
    executor = CountingExecutor(output_limit_executor())

    first = _run([request], executor=executor, cache=cache)
    second = _run([request], executor=executor, cache=cache)

    assert executor.call_count == 1
    assert second == first
    assert is_output_limit_outcome(second[request])
    assert second[request].stdout == ""


def test_run_requests_caches_killed_outcome() -> None:
    from dr_code.metrics.engine.execution import (
        InMemoryExecutionCache,
        is_killed_outcome,
    )

    request = _request()
    cache = InMemoryExecutionCache()
    executor = CountingExecutor(
        scripted_executor(returncode=-9, stderr="killed")
    )

    first = _run([request], executor=executor, cache=cache)
    second = _run([request], executor=executor, cache=cache)

    assert executor.call_count == 1
    assert second == first
    assert is_killed_outcome(second[request])
    assert "killed" in second[request].stderr


def test_planning_executor_failure_propagates_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    counting_executor,
) -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    operator_cls = REGISTRY[str(MetricName.TEXT_STATS)]

    def fail_planning(self, value, aux):  # noqa: ANN001
        raise ExecutorFailure("planning infrastructure failed")

    monkeypatch.setattr(operator_cls, "execution_requests", fail_planning)

    with pytest.raises(ExecutorFailure):
        _extract(
            _definition([_q("text_stats")]),
            trace,
            executor=counting_executor,
        )
    assert counting_executor.call_count == 0


def test_infrastructure_executor_failure_raises(
    task, code_test_trace, raising
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(ExecutorFailure):
        _extract(
            definition,
            trace,
            executor=raising(ExecutorFailure("infra broke")),
        )


def test_missing_execution_outcome_raises_engine_invariant_error(
    task,
    local_executor,
    code_test_trace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics import EngineInvariantError
    from dr_code.humaneval.metric_operator import CodeTest

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    def no_requests(self, value, aux):  # noqa: ANN001
        return ()

    monkeypatch.setattr(CodeTest, "execution_requests", no_requests)
    with pytest.raises(EngineInvariantError):
        _extract(definition, trace, executor=local_executor)


def test_wall_time_budget_is_candidate_data_not_infrastructure(
    task, code_test_trace
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=1.0)]
    )
    record = _extract(
        definition,
        trace,
        executor=timeout_executor(),
    )[0]
    assert record.status.value == "measured"
    facts = _facts(record)
    assert facts["timeout_count"] == facts["total_cases"]


def test_batch_dedupes_identical_code_test_executions(
    task, counting_executor, code_test_trace
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(
        definition, [trace, trace, trace], executor=counting_executor
    )
    assert counting_executor.call_count == 1


def test_distinct_submissions_execute_separately(
    task, counting_executor, code_test_trace
) -> None:
    good = code_test_trace("def add_one(x):\n    return x + 1\n", task)
    bad = code_test_trace("def add_one(x):\n    return x - 1\n", task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(definition, [good, bad], executor=counting_executor)
    assert counting_executor.call_count == 2


def test_batch_returns_one_record_tuple_per_trace(
    task, local_executor, code_test_trace
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=5.0)]
    )
    results = _extract_batch(
        definition, [trace, trace], executor=local_executor
    )
    assert isinstance(results, tuple)
    assert len(results) == 2
    for per_trace in results:
        assert isinstance(per_trace, tuple)
        assert len(per_trace) == 1


def test_prepopulated_execution_cache_skips_the_executor(
    task, counting_executor, code_test_trace
) -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    cache = InMemoryExecutionCache()
    _extract(
        definition,
        trace,
        executor=counting_executor,
        execution_cache=cache,
    )
    assert counting_executor.call_count == 1

    counting_executor.calls.clear()
    _extract(
        definition,
        trace,
        executor=counting_executor,
        execution_cache=cache,
    )
    assert counting_executor.call_count == 0


def test_pure_operators_never_call_the_executor(counting_executor) -> None:
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
    _extract_batch(definition, [trace, trace], executor=counting_executor)
    assert counting_executor.call_count == 0


def test_pure_operators_need_no_executor() -> None:
    text = "def f(x):\n    return x\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=text),
            "output": CodeArtifact(source=text),
        }
    )
    definition = _definition([_q("text_stats", on="input")])
    records = _extract(definition, trace)
    assert records[0].status.value == "measured"


def test_code_test_record_values_exclude_timing(
    task, local_executor, code_test_trace
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=2.0)]
    )
    record = _extract(definition, trace, executor=local_executor)[0]
    assert "elapsed_seconds" not in _facts(record)
