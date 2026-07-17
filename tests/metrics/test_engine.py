"""Engine contracts (plan section: ``engine/engine.py``).

Covers the four engine promises (design L3) and the bind/plan/compute flow:

* bind-time ``WiringError`` on incompatible definitions, before any work;
* totality — N questions ⇒ N records, in declaration order;
* Absent ``on``/aux input ⇒ NOT_APPLICABLE record preserving the cause
  (missing key is still a wiring error — design L2);
* operator exception ⇒ OPERATOR_FAILURE record attributed to the metric;
* infrastructure ``SandboxError`` raises (fail-closed);
* candidate timeout is data, not infrastructure;
* two-phase execution with content-hash request dedupe (X-S4);
* determinism across fresh / deserialized / external traces (X-S2).

``dr_code.metrics`` is imported lazily inside each test. All execution goes
through the injectable ``SandboxRunner`` seam — never a real container.
"""

from __future__ import annotations

import pytest

from dr_code.humaneval.sandbox import SandboxError, SandboxTimeoutError
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

from metrics.helpers import code_test_trace, raising_runner


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

    return MetricQuestion(metric=MetricName(metric_name), on=on, settings=settings)


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
        {"input": TextArtifact(text="not code"), "output": TextArtifact(text="x")}
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_invalid_operator_settings_is_a_wiring_error(counting_runner) -> None:
    """compressed_length requires an explicit method+level; bad settings wire."""
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition([_q("compressed_length", method="gzip")])  # no level
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_missing_auxiliary_key_is_a_wiring_error(task, counting_runner) -> None:
    """code_test needs its task key; a key absent from the namespace is wiring."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = external_trace(
        {"input": CodeArtifact(source=candidate), "output": CodeArtifact(source=candidate)}
    )
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_batch_wiring_error_runs_no_sandbox_work(task, counting_runner) -> None:
    bad = external_trace(
        {"input": TextArtifact(text="not code"), "output": TextArtifact(text="x")}
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
        {"input": CodeArtifact(source=text), "output": CodeArtifact(source=text)}
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
    """Absent input ⇒ NOT_APPLICABLE carrying the Absent lineage (design L3),
    distinct from a missing key (which is a wiring error)."""
    trace = external_trace(
        {
            "input": Absent(failed_step="extract", cause="no code"),
            "output": Absent(failed_step="extract", cause="no code"),
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
    wiring bug (plan: Absent aux values yield not-applicable records)."""
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
    record = _extract(definition, trace)[0]
    assert record.status.value == "not_applicable"
    assert record.absence_failed_step == "load"


# ===========================================================================
# Operator exception ⇒ OPERATOR_FAILURE record (totality, L3).
# ===========================================================================

def test_operator_exception_becomes_an_operator_failure_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator bug on present input is a visible failure record, never a
    boundary-crossing exception (design L3)."""
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    text = "def f(x):\n    return x + 1\n"
    trace = external_trace(
        {"input": CodeArtifact(source=text), "output": CodeArtifact(source=text)}
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


def test_ast_stats_raises_on_unparseable_code_instead_of_fabricating_zeros() -> None:
    """CodeArtifact documents "passed a compile check upstream", so unparseable
    CODE is a producer contract violation -- ast_stats must not mask it as an
    all-zero (indistinguishable from empty) measurement. It becomes an
    OPERATOR_FAILURE record, consistent with code_test's SyntaxError-on-parse
    behavior; parse facts stay the job of parse_outcome."""
    from dr_code.metrics import MetricName

    invalid = "def f(:\n    pass\n"
    trace = external_trace(
        {"input": CodeArtifact(source=invalid), "output": CodeArtifact(source=invalid)}
    )
    definition = _definition([_q("ast_stats", on="input")])
    record = _extract(definition, trace)[0]
    assert record.status.value == "operator_failure"
    assert record.metric is MetricName.AST_STATS
    assert record.values == {}


# ===========================================================================
# Infrastructure SandboxError raises; candidate timeout is data (L3).
# ===========================================================================

def test_infrastructure_sandbox_error_raises(task) -> None:
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


def test_sandbox_timeout_is_candidate_data_not_infrastructure(task) -> None:
    """A candidate timeout is attributed to the candidate as data (timeout
    cases), not a raised SandboxError (batch_runner attribution parity)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input", timeout_seconds=1.0)])
    record = _extract(
        definition,
        trace,
        run_in_sandbox=raising_runner(SandboxTimeoutError("timed out")),
    )[0]
    assert record.status.value == "measured"
    assert record.values["timeout_count"] == record.values["total_cases"]


# ===========================================================================
# Two-phase execution + content-hash request dedupe (X-S4).
# ===========================================================================

def test_batch_dedupes_identical_code_test_executions(
    task, counting_runner
) -> None:
    """Identical submissions across a sweep execute once (at-most-once)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(definition, [trace, trace, trace], run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 1


def test_distinct_submissions_execute_separately(task, counting_runner) -> None:
    good = code_test_trace("def add_one(x):\n    return x + 1\n", task)
    bad = code_test_trace("def add_one(x):\n    return x - 1\n", task)
    definition = _definition([_q("code_test", on="input")])
    _extract_batch(definition, [good, bad], run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 2


def test_batch_returns_one_record_tuple_per_trace(task, local_runner) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input", timeout_seconds=5.0)])
    results = _extract_batch(definition, [trace, trace], run_in_sandbox=local_runner)
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
    _extract(definition, trace, run_in_sandbox=counting_runner, execution_cache=cache)
    assert counting_runner.call_count == 1

    counting_runner.calls.clear()
    _extract(definition, trace, run_in_sandbox=counting_runner, execution_cache=cache)
    assert counting_runner.call_count == 0


def test_pure_operators_never_call_the_runner(counting_runner) -> None:
    """Pure operators declare no execution requests (X-M4)."""
    text = "def f(x):\n    return x\n"
    trace = external_trace(
        {"input": CodeArtifact(source=text), "output": CodeArtifact(source=text)}
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )
    _extract_batch(definition, [trace, trace], run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


# ===========================================================================
# Record equality across fresh / deserialized / external traces (X-S2).
# ===========================================================================

def test_fresh_trace_equals_deserialized_trace() -> None:
    """Restored traces measure the same as fresh traces (design L2/L3)."""
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
    """Any producer's trace yields the same measured answer (X-S2). Producer
    lineage legitimately differs, so the comparable projection is the answer."""
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

    assert [answer(r) for r in _extract(definition, external)] == [
        answer(r) for r in _extract(definition, preprocessing)
    ]


def test_code_test_record_values_exclude_timing(task, local_runner) -> None:
    """Determinism soft spot: timing stays out of record values so identical
    inputs reproduce (plan section 3)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition([_q("code_test", on="input", timeout_seconds=2.0)])
    record = _extract(definition, trace, run_in_sandbox=local_runner)[0]
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
