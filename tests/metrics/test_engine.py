"""Engine contracts (plan section: ``engine/engine.py``).

Covers the four engine promises and the bind/plan/compute flow:

* bind-time ``WiringError`` on incompatible definitions, before any work;
* totality — N questions ⇒ N records, in declaration order;
* Absent ``on``/aux input ⇒ NOT_APPLICABLE record preserving the cause
  (missing key is still a wiring error);
* operator exception ⇒ OPERATOR_FAILURE record attributed to the metric;
* a genuine executor failure (``ExecutorFailure``) raises (fail-closed);
* candidate budget deaths are data, not infrastructure;
* two-phase execution with content-hash request dedupe (X-S4);
* determinism across fresh / deserialized / external traces (X-S2).

``dr_code.metrics`` is imported lazily inside each test. Logic tests drive a
scripted ``FakeExecutor``; parity tests drive the real batch executor.
"""

from __future__ import annotations

import pytest
from dr_exec import ExecutorFailure
from dr_code.trace import (
    Absent,
    CodeArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    WiringError,
    deserialize_trace,
    external_trace,
    serialize_trace,
)

from metrics.helpers import (
    PRODUCTION_EXECUTOR,
    code_test_trace,
    fake_executor_always,
    full_pass_batch,
    output_budget_run,
    scripted_batch,
    wall_clock_run,
)


# ---------------------------------------------------------------------------
# Definition helpers (lazy metrics imports inside).
# ---------------------------------------------------------------------------


def _definition(questions) -> object:
    from dr_code.metrics import MetricsDefinition

    return MetricsDefinition(
        definition_id="def", version="1", questions=tuple(questions)
    )


def _q(metric_name: str, on: str = "input", **settings) -> object:
    from dr_code.metrics import MetricName, MetricQuestion

    return MetricQuestion(
        metric=MetricName(metric_name), on=on, settings=settings
    )


def _pass_all_executor():
    """A fake that answers every batch by passing every requested case."""

    def batch_for(call):
        case_ids = call.request.item_ids
        return full_pass_batch(case_ids=case_ids)

    return fake_executor_always(batch_for)


# ===========================================================================
# Bind-time WiringError, before any work.
# ===========================================================================


def test_missing_on_key_is_a_wiring_error_before_any_work(
    counting_executor,
) -> None:
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition([_q("text_stats", on="nonexistent")])
    with pytest.raises(WiringError):
        _extract(definition, trace, executor=counting_executor)
    assert counting_executor.call_count == 0


def test_wrong_artifact_kind_is_a_wiring_error(counting_executor) -> None:
    trace = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, executor=counting_executor)
    assert counting_executor.call_count == 0


def test_invalid_operator_settings_is_a_wiring_error(
    counting_executor,
) -> None:
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition(
        [_q("compressed_length", compression={"method": "gzip", "level": 99})]
    )
    with pytest.raises(WiringError):
        _extract(definition, trace, executor=counting_executor)
    assert counting_executor.call_count == 0


def test_missing_auxiliary_key_is_a_wiring_error(
    task, counting_executor
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=candidate),
            "output": CodeArtifact(source=candidate),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, executor=counting_executor)
    assert counting_executor.call_count == 0


def test_batch_wiring_error_runs_no_execution_work(
    task, counting_executor
) -> None:
    bad = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract_batch(definition, [bad, bad], executor=counting_executor)
    assert counting_executor.call_count == 0


# ===========================================================================
# Totality — one record per declared question, in declaration order.
# ===========================================================================


def test_one_record_per_question_in_declaration_order() -> None:
    text = "def f(x):\n    return x + 1\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=text),
            "output": CodeArtifact(source=text),
        }
    )
    definition = _definition(
        [
            _q("text_stats", on="input"),
            _q("code_leakage", on="input", task_names=["f"]),
            _q("ast_stats", on="input"),
        ]
    )
    records = _extract(definition, trace, executor=PRODUCTION_EXECUTOR)
    assert len(records) == 3
    assert [r.metric.value for r in records] == [
        "text_stats",
        "code_leakage",
        "ast_stats",
    ]


def test_no_questions_yields_no_records() -> None:
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    assert _extract(_definition([]), trace, executor=PRODUCTION_EXECUTOR) == ()


def test_absent_on_key_yields_not_applicable_with_cause() -> None:
    trace = external_trace(
        {
            "input": Absent(failed_step="extract", cause="no code"),
            "output": Absent(failed_step="extract", cause="no code"),
        }
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )
    records = _extract(definition, trace, executor=PRODUCTION_EXECUTOR)
    assert len(records) == 2
    for record in records:
        assert record.status.value == "not_applicable"
        assert record.absence_failed_step == "extract"
        assert record.absence_cause == "no code"
        assert record.values == {}


def test_absent_auxiliary_yields_not_applicable(task) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    code = CodeArtifact(source=candidate)
    trace = external_trace(
        {
            "input": code,
            "output": code,
            "task": Absent(failed_step="load", cause="missing task"),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    record = _extract(definition, trace, executor=PRODUCTION_EXECUTOR)[0]
    assert record.status.value == "not_applicable"
    assert record.absence_failed_step == "load"


# ===========================================================================
# Operator exception ⇒ OPERATOR_FAILURE record (totality).
# ===========================================================================


def test_operator_exception_becomes_an_operator_failure_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    text = "def f(x):\n    return x + 1\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=text),
            "output": CodeArtifact(source=text),
        }
    )

    operator_cls = REGISTRY[str(MetricName.TEXT_STATS)]

    def boom(self, value, aux, ctx):  # noqa: ANN001
        raise ValueError("operator bug")

    monkeypatch.setattr(operator_cls, "compute", boom)
    definition = _definition([_q("text_stats", on="input")])
    record = _extract(definition, trace, executor=PRODUCTION_EXECUTOR)[0]
    assert record.status.value == "operator_failure"
    assert record.failure_type == "ValueError"
    assert record.failure_message == "operator bug"
    assert record.metric is MetricName.TEXT_STATS


def test_ast_stats_raises_on_unparseable_code_instead_of_fabricating_zeros() -> (
    None
):
    from dr_code.metrics import MetricName

    invalid = "def f(:\n    pass\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=invalid),
            "output": CodeArtifact(source=invalid),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    record = _extract(definition, trace, executor=PRODUCTION_EXECUTOR)[0]
    assert record.status.value == "operator_failure"
    assert record.metric is MetricName.AST_STATS
    assert record.values == {}


# ===========================================================================
# Executor failures raise; candidate budget deaths are data.
# ===========================================================================


def test_executor_failure_raises(task) -> None:
    """A genuine ExecutorFailure (no result to attribute) raises — never a
    record. It is the only propagating infrastructure path."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    def raise_failure(call):
        raise ExecutorFailure("executor broke")

    with pytest.raises(ExecutorFailure):
        _extract(
            definition, trace, executor=fake_executor_always(raise_failure)
        )


def test_missing_execution_outcome_raises_engine_invariant_error(
    task, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compute that plans a request execution_requests never saw is an engine
    bug: it surfaces as EngineInvariantError, never an operator_failure."""
    from dr_code.metrics import EngineInvariantError
    from dr_code.metrics.operators.code_test import CodeTest

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    def no_requests(self, value, aux):  # noqa: ANN001
        return ()

    monkeypatch.setattr(CodeTest, "execution_requests", no_requests)
    with pytest.raises(EngineInvariantError):
        _extract(definition, trace, executor=PRODUCTION_EXECUTOR)


def test_wall_clock_budget_is_candidate_data_not_infrastructure(task) -> None:
    """A wall-clock budget death is scored against the candidate as timeout
    cases, not a raised failure."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=1.0)]
    )

    def timed_out(call):
        return scripted_batch(case_payloads={}, run=wall_clock_run(None))

    record = _extract(
        definition, trace, executor=fake_executor_always(timed_out)
    )[0]
    assert record.status.value == "measured"
    assert record.values["timeout_count"] == record.values["total_cases"]


def test_output_budget_is_candidate_data_not_infrastructure(task) -> None:
    """Candidate output flooding becomes error cases, not an infra raise."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    def flooded(call):
        return scripted_batch(case_payloads={}, run=output_budget_run())

    record = _extract(
        definition, trace, executor=fake_executor_always(flooded)
    )[0]
    assert record.status.value == "measured"
    assert record.values["error_count"] == record.values["total_cases"]
    assert record.values["timeout_count"] == 0


# ===========================================================================
# Two-phase execution + content-hash request dedupe (X-S4).
# ===========================================================================


def test_batch_dedupes_identical_code_test_executions(
    task, counting_executor
) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(
        definition, [trace, trace, trace], executor=counting_executor
    )
    assert counting_executor.call_count == 1


def test_distinct_submissions_execute_separately(
    task, counting_executor
) -> None:
    good = code_test_trace("def add_one(x):\n    return x + 1\n", task)
    bad = code_test_trace("def add_one(x):\n    return x - 1\n", task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(definition, [good, bad], executor=counting_executor)
    assert counting_executor.call_count == 2


def test_batch_returns_one_record_tuple_per_trace(task) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=5.0)]
    )
    results = _extract_batch(
        definition, [trace, trace], executor=PRODUCTION_EXECUTOR
    )
    assert isinstance(results, tuple)
    assert len(results) == 2
    for per_trace in results:
        assert isinstance(per_trace, tuple)
        assert len(per_trace) == 1


def test_prepopulated_execution_cache_skips_the_executor(
    task, counting_executor
) -> None:
    from dr_code.metrics.engine.execution import InMemoryExecutionCache

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    cache = InMemoryExecutionCache()
    _extract(
        definition, trace, executor=counting_executor, execution_cache=cache
    )
    assert counting_executor.call_count == 1

    counting_executor.calls.clear()
    _extract(
        definition, trace, executor=counting_executor, execution_cache=cache
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


# ===========================================================================
# Record equality across fresh / deserialized / external traces (X-S2).
# ===========================================================================


def test_fresh_trace_equals_deserialized_trace() -> None:
    text = "def add_one(x):\n    return x + 1\n"
    fresh = external_trace(
        {
            "input": CodeArtifact(source=text),
            "output": CodeArtifact(source=text),
        }
    )
    restored = deserialize_trace(serialize_trace(fresh))
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )
    assert _extract(
        definition, fresh, executor=PRODUCTION_EXECUTOR
    ) == _extract(definition, restored, executor=PRODUCTION_EXECUTOR)


def test_external_trace_matches_preprocessing_producer_trace() -> None:
    text = "def f(x):\n    return x\n"
    values = {
        "input": CodeArtifact(source=text),
        "output": CodeArtifact(source=text),
    }
    external = external_trace(values)
    preprocessing = Trace(
        values=values,
        producer=TraceProducer(
            producer_id="pre", version="v1", definition_hash="abc"
        ),
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )

    def answer(record):
        return (
            record.metric,
            record.metric_version,
            record.on_key,
            record.status,
            tuple(sorted(record.values.items())),
        )

    assert [
        answer(r)
        for r in _extract(definition, external, executor=PRODUCTION_EXECUTOR)
    ] == [
        answer(r)
        for r in _extract(
            definition, preprocessing, executor=PRODUCTION_EXECUTOR
        )
    ]


def test_code_test_record_values_exclude_timing(task) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=2.0)]
    )
    record = _extract(definition, trace, executor=PRODUCTION_EXECUTOR)[0]
    assert "elapsed_seconds" not in record.values


# ---------------------------------------------------------------------------
# Engine call wrappers (keep the lazy import in one place).
# ---------------------------------------------------------------------------


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return extract_metrics(definition, trace, **kwargs)


def _extract_batch(definition, traces, **kwargs):
    from dr_code.metrics import extract_metrics_batch

    return extract_metrics_batch(definition, traces, **kwargs)
