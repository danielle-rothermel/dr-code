"""Metrics-engine contracts.

Covers the bind, plan, execute, and compute flow:

* bind-time ``WiringError`` on incompatible definitions, before any work;
* totality — N questions ⇒ N records, in declaration order;
* Absent ``on``/aux input ⇒ NOT_APPLICABLE record preserving the cause
  (a missing key remains a wiring error);
* operator exception ⇒ OPERATOR_FAILURE record attributed to the metric;
* infrastructure ``SandboxError`` raises (fail-closed);
* candidate timeout is data, not infrastructure;
* two-phase execution with equivalent-request deduplication.

All execution goes through the injectable ``SandboxRunner`` seam.
"""

from __future__ import annotations

import pytest

from dr_code.core.execution.sandbox import SandboxError, SandboxTimeoutError
from dr_code.trace import (
    Absent,
    CodeArtifact,
    TextArtifact,
    WiringError,
    external_trace,
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
        _extract(definition, trace, run_in_sandbox=counting_runner)
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
        _extract(definition, trace, run_in_sandbox=counting_runner)
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

    assert record.identity.metric_version == "test-version"


def test_records_preserve_complete_metrics_definition_coordinates() -> None:
    from dr_code.metrics import (
        METRIC_RECORD_ADAPTER,
        MetricsDefinitionCoordinate,
    )

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
        baseline_record.identity.metrics_definition,
        reordered_record.identity.metrics_definition,
        changed_record.identity.metrics_definition,
    } == {
        MetricsDefinitionCoordinate.of(baseline),
        MetricsDefinitionCoordinate.of(reordered),
        MetricsDefinitionCoordinate.of(changed_settings),
    }

    restored = METRIC_RECORD_ADAPTER.validate_json(
        changed_record.model_dump_json()
    )
    assert restored == changed_record
    assert restored.identity.metrics_definition == (
        MetricsDefinitionCoordinate.of(changed_settings)
    )


def test_invalid_operator_settings_fail_at_definition_boundary() -> None:
    """Invalid settings cannot enter a metrics definition."""
    with pytest.raises(Exception):
        _q(
            "compressed_length",
            compression={"method": "gzip", "level": 99},
        )


def test_missing_auxiliary_key_is_a_wiring_error(counting_runner) -> None:
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
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_batch_wiring_error_runs_no_sandbox_work(counting_runner) -> None:
    bad = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract_batch(definition, [bad, bad], run_in_sandbox=counting_runner)
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
    assert [r.identity.question.metric.value for r in records] == [
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
                failure_code="no_alternative_produced_candidates",
                cause="no code",
            ),
            "output": Absent(
                failed_step="extract",
                failure_code="no_alternative_produced_candidates",
                cause="no code",
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
        assert record.absence.failed_step == "extract"
        assert record.absence.failure_code == (
            "no_alternative_produced_candidates"
        )
        assert record.absence.cause == "no code"


def test_absent_auxiliary_yields_not_applicable() -> None:
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
                failure_code="missing_task",
                cause="missing task",
            ),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    record = _extract(definition, trace)[0]
    assert record.status.value == "not_applicable"
    assert record.absence.failed_step == "load"
    assert record.absence.failure_code == "missing_task"


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
    assert record.failure.failure_type == "ValueError"
    assert record.failure.failure_message == "operator bug"
    assert record.identity.question.metric is MetricName.TEXT_STATS


def test_operator_result_violating_a_record_invariant_is_a_failure_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator that returns a result no MeasuredRecord can hold -- here a
    zero-fact result -- is a misbehaving operator like any other, so it becomes
    an operator-failure record for that one question rather than aborting the
    whole batch with a ValidationError."""
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

    class _NoFacts:
        def to_facts(self) -> tuple[()]:
            return ()

    def empty(self, value, aux, ctx):  # noqa: ANN001
        return _NoFacts()

    monkeypatch.setattr(operator_cls, "compute", empty)
    # A second question shares the batch: the misbehaving operator must not
    # take the well-behaved one down with it.
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )

    records = _extract(definition, trace)

    by_metric = {record.identity.question.metric: record for record in records}
    failed = by_metric[MetricName.TEXT_STATS]
    assert failed.status.value == "operator_failure"
    assert failed.failure.failure_type == "ValidationError"
    assert by_metric[MetricName.AST_STATS].status.value == "measured"


def test_ast_stats_raises_on_unparseable_code_instead_of_fabricating_zeros() -> (
    None
):
    """CodeArtifact documents "passed a compile check upstream", so unparseable
    CODE is a producer contract violation -- ast_stats must not mask it as an
    all-zero (indistinguishable from empty) measurement. It becomes an
    OPERATOR_FAILURE record, consistent with code_test's SyntaxError-on-parse
    behavior; parse facts stay the job of parse_outcome."""
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
    assert record.identity.question.metric is MetricName.AST_STATS
    assert not hasattr(record, "facts")


# ===========================================================================
# Infrastructure SandboxError raises; candidate timeout is data.
# ===========================================================================


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


# ===========================================================================
# Two-phase execution and equivalent-request deduplication.
# ===========================================================================


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


# ---------------------------------------------------------------------------
# Engine call wrappers.
# ---------------------------------------------------------------------------


def _facts(record):
    """The measured record's facts as a name-to-value mapping."""
    assert record.status.value == "measured", record
    return {fact.name: fact.value for fact in record.facts}


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return extract_metrics(definition, trace, **kwargs)


def _extract_batch(definition, traces, **kwargs):
    from dr_code.metrics import extract_metrics_batch

    return extract_metrics_batch(definition, traces, **kwargs)
