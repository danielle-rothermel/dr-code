"""Metrics-engine contracts.

Covers the bind, plan, execute, and compute flow:

* bind-time ``WiringError`` on incompatible definitions, before any work;
* totality — N questions ⇒ N records, in declaration order;
* Absent ``on``/aux input ⇒ NOT_APPLICABLE record preserving the cause
  (a missing key remains a wiring error);
* operator exception ⇒ OPERATOR_FAILURE record attributed to the metric;
* infrastructure ``SubprocessError`` raises (fail-closed);
* candidate timeout is data, not infrastructure;
* two-phase execution with equivalent-request deduplication;
* equal answers across fresh, deserialized, and external traces.

All execution goes through the injectable ``PythonSubprocessRunner`` seam.
"""

from __future__ import annotations

import pytest

from dr_code.execution.subprocess import (
    SubprocessError,
    SubprocessOutputLimitError,
    SubprocessTimeoutError,
)
from dr_code.trace import (
    Absent,
    CodeArtifact,
    ComponentCoordinate,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    StepCoordinate,
    TextArtifact,
    Trace,
    WiringError,
    deserialize_trace,
    external_trace,
    serialize_trace,
)

from metrics.helpers import code_test_trace, raising_runner


def _preprocessing_producer() -> PreprocessingTraceProducer:
    return PreprocessingTraceProducer(
        definition=PreprocessingDefinitionCoordinate(
            definition_id="pre",
            version="test-version",
            steps=(
                StepCoordinate(
                    instance_name="step",
                    component=ComponentCoordinate(
                        registered_name="normalize_unicode",
                        version="0",
                    ),
                ),
            ),
        )
    )


# ---------------------------------------------------------------------------
# Definition helpers.
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


# ===========================================================================
# Bind-time WiringError, before any work.
# ===========================================================================


def test_missing_on_key_is_a_wiring_error_before_any_work(
    counting_runner,
) -> None:
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition([_q("text_stats", on="nonexistent")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_subprocess=counting_runner)
    assert counting_runner.call_count == 0


def test_wrong_artifact_kind_is_a_wiring_error(counting_runner) -> None:
    """ast_stats requires CODE; a TEXT key is a kind mismatch (bind-time)."""
    trace = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_subprocess=counting_runner)
    assert counting_runner.call_count == 0


def test_record_reads_operator_declared_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics.registry import REGISTRY

    operator = REGISTRY["text_stats"]
    monkeypatch.setattr(operator, "VERSION", "test-version")
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )

    record = _extract(_definition([_q("text_stats")]), trace)[0]

    assert record.metric_version == "test-version"


def test_records_preserve_complete_metrics_definition_coordinates() -> None:
    from dr_code.metrics import MetricRecord

    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    text_stats = _q("text_stats")
    leakage_a = _q("code_leakage", task_names=["a"])
    leakage_b = _q("code_leakage", task_names=["b"])
    baseline = _definition([text_stats, leakage_a])
    reordered = _definition([leakage_a, text_stats])
    changed_settings = _definition([text_stats, leakage_b])

    baseline_record = _extract(baseline, trace)[0]
    reordered_record = _extract(reordered, trace)[0]
    changed_record = _extract(changed_settings, trace)[0]

    assert baseline.definition_id == reordered.definition_id
    assert baseline.version == reordered.version
    assert {
        baseline_record.metrics_definition,
        reordered_record.metrics_definition,
        changed_record.metrics_definition,
    } == {baseline, reordered, changed_settings}

    restored = MetricRecord.model_validate_json(
        changed_record.model_dump_json()
    )
    assert restored == changed_record
    assert restored.metrics_definition == changed_settings


def test_invalid_operator_settings_fail_at_definition_boundary(
    counting_runner,
) -> None:
    """Invalid settings cannot enter a metrics definition."""
    with pytest.raises(Exception):
        _q(
            "compressed_length",
            compression={"method": "gzip", "level": 99},
        )
    assert counting_runner.call_count == 0


def test_missing_auxiliary_key_is_a_wiring_error(
    task, counting_runner
) -> None:
    """code_test needs its task key; a key absent from the namespace is wiring."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=candidate),
            "output": CodeArtifact(source=candidate),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_subprocess=counting_runner)
    assert counting_runner.call_count == 0


def test_batch_wiring_error_runs_no_subprocess_work(
    task, counting_runner
) -> None:
    bad = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract_batch(
            definition, [bad, bad], run_in_subprocess=counting_runner
        )
    assert counting_runner.call_count == 0


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
    records = _extract(definition, trace)
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
    assert _extract(_definition([]), trace) == ()


def test_absent_on_key_yields_not_applicable_with_cause() -> None:
    """Absent input ⇒ NOT_APPLICABLE carrying the Absent lineage,
    distinct from a missing key (which is a wiring error)."""
    trace = external_trace(
        {
            "input": Absent(
                failed_step="extract",
                cause="no code",
                failure_code="test.no_code",
            ),
            "output": Absent(
                failed_step="extract",
                cause="no code",
                failure_code="test.no_code",
            ),
        }
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )
    records = _extract(definition, trace)
    assert len(records) == 2
    for record in records:
        assert record.status.value == "not_applicable"
        assert record.absence_failed_step == "extract"
        assert record.absence_cause == "no code"
        assert record.values == {}


def test_absent_auxiliary_yields_not_applicable(task) -> None:
    """code_test whose task aux is present-but-Absent is not-applicable, not a
    wiring bug."""
    candidate = "def add_one(x):\n    return x + 1\n"
    code = CodeArtifact(source=candidate)
    trace = external_trace(
        {
            "input": code,
            "output": code,
            "task": Absent(
                failed_step="load",
                cause="missing task",
                failure_code="test.missing_task",
            ),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    record = _extract(definition, trace)[0]
    assert record.status.value == "not_applicable"
    assert record.absence_failed_step == "load"


# ===========================================================================
# Operator exception ⇒ OPERATOR_FAILURE record.
# ===========================================================================


def test_operator_exception_becomes_an_operator_failure_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator bug on present input is a visible failure record, never a
    boundary-crossing exception."""
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
    record = _extract(definition, trace)[0]
    assert record.status.value == "operator_failure"
    assert record.failure_type == "ValueError"
    assert record.failure_message == "operator bug"
    assert record.metric is MetricName.TEXT_STATS


def test_ast_stats_raises_on_unparseable_code_instead_of_fabricating_zeros() -> (
    None
):
    """Unparseable code becomes an operator failure, not an all-zero result.

    ``CodeArtifact`` itself carries no compilation guarantee. ``ast_stats``
    therefore owns parsing failure at its consumer boundary, while explicit
    parse facts remain the job of ``parse_outcome``.
    """
    from dr_code.metrics import MetricName

    invalid = "def f(:\n    pass\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=invalid),
            "output": CodeArtifact(source=invalid),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    record = _extract(definition, trace)[0]
    assert record.status.value == "operator_failure"
    assert record.metric is MetricName.AST_STATS
    assert record.values == {}


# ===========================================================================
# Infrastructure errors raise; candidate terminations are data.
# ===========================================================================


def test_infrastructure_subprocess_error_raises(task) -> None:
    """A SubprocessError is infra breakage — it raises, never becomes a record."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(SubprocessError):
        _extract(
            definition,
            trace,
            run_in_subprocess=raising_runner(SubprocessError("infra broke")),
        )


def test_missing_execution_outcome_raises_engine_invariant_error(
    task, local_runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If CodeTest.compute rebuilds an ExecutionRequest that diverges from
    what execution_requests planned, ctx.outcome_for's lookup misses. That
    is an engine bug, not a metric bug: it must surface as
    EngineInvariantError out of the batch, never get swallowed into an
    operator_failure record."""
    from dr_code.metrics import EngineInvariantError
    from dr_code.metrics.operators.code_test import CodeTest

    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])

    def no_requests(self, value, aux):  # noqa: ANN001
        return ()

    monkeypatch.setattr(CodeTest, "execution_requests", no_requests)
    with pytest.raises(EngineInvariantError):
        _extract(definition, trace, run_in_subprocess=local_runner)


def test_subprocess_timeout_is_candidate_data_not_infrastructure(task) -> None:
    """A candidate timeout is attributed to the candidate as data (timeout
    cases), not a raised SubprocessError (batch_runner attribution parity)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=1.0)]
    )
    record = _extract(
        definition,
        trace,
        run_in_subprocess=raising_runner(SubprocessTimeoutError("timed out")),
    )[0]
    assert record.status.value == "measured"
    assert record.values["timeout_count"] == record.values["total_cases"]


def test_subprocess_output_limit_is_candidate_data_not_infrastructure(
    task,
) -> None:
    """Candidate output flooding becomes error cases, not an infra raise."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    record = _extract(
        definition,
        trace,
        run_in_subprocess=raising_runner(
            SubprocessOutputLimitError("too much output")
        ),
    )[0]
    assert record.status.value == "measured"
    assert record.values["error_count"] == record.values["total_cases"]
    assert record.values["timeout_count"] == 0


# ===========================================================================
# Two-phase execution and equivalent-request deduplication.
# ===========================================================================


def test_batch_dedupes_identical_code_test_executions(
    task, counting_runner
) -> None:
    """Identical submissions across a sweep execute once (at-most-once)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(
        definition, [trace, trace, trace], run_in_subprocess=counting_runner
    )
    assert counting_runner.call_count == 1


def test_distinct_submissions_execute_separately(
    task, counting_runner
) -> None:
    good = code_test_trace("def add_one(x):\n    return x + 1\n", task)
    bad = code_test_trace("def add_one(x):\n    return x - 1\n", task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(definition, [good, bad], run_in_subprocess=counting_runner)
    assert counting_runner.call_count == 2


def test_batch_returns_one_record_tuple_per_trace(task, local_runner) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=5.0)]
    )
    results = _extract_batch(
        definition, [trace, trace], run_in_subprocess=local_runner
    )
    assert isinstance(results, tuple)
    assert len(results) == 2
    for per_trace in results:
        assert isinstance(per_trace, tuple)
        assert len(per_trace) == 1


def test_prepopulated_execution_cache_skips_the_runner(
    task, counting_runner
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
        run_in_subprocess=counting_runner,
        execution_cache=cache,
    )
    assert counting_runner.call_count == 1

    counting_runner.calls.clear()
    _extract(
        definition,
        trace,
        run_in_subprocess=counting_runner,
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
    _extract_batch(
        definition, [trace, trace], run_in_subprocess=counting_runner
    )
    assert counting_runner.call_count == 0


# ===========================================================================
# Record equality across fresh, deserialized, and external traces.
# ===========================================================================


def test_fresh_trace_equals_deserialized_trace() -> None:
    """Restored traces measure the same as fresh traces."""
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
    assert _extract(definition, fresh) == _extract(definition, restored)


def test_external_trace_matches_preprocessing_producer_trace() -> None:
    """Any producer's trace yields the same measured answer. Producer
    lineage legitimately differs, so the comparable projection is the answer."""
    text = "def f(x):\n    return x\n"
    values = {
        "input": CodeArtifact(source=text),
        "output": CodeArtifact(source=text),
    }
    external = external_trace(values)
    preprocessing = Trace(
        values=values,
        producer=_preprocessing_producer(),
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

    assert [answer(r) for r in _extract(definition, external)] == [
        answer(r) for r in _extract(definition, preprocessing)
    ]


def test_code_test_record_values_exclude_timing(task, local_runner) -> None:
    """Determinism soft spot: timing stays out of record values so identical
    inputs reproduce."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=2.0)]
    )
    record = _extract(definition, trace, run_in_subprocess=local_runner)[0]
    assert "elapsed_seconds" not in record.values


# ---------------------------------------------------------------------------
# Engine call wrappers.
# ---------------------------------------------------------------------------


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return extract_metrics(definition, trace, **kwargs)


def _extract_batch(definition, traces, **kwargs):
    from dr_code.metrics import extract_metrics_batch

    return extract_metrics_batch(definition, traces, **kwargs)
