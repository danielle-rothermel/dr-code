"""Engine contracts (plan section: ``engine/engine.py``).

Covers the four engine promises (design L3) and the bind/plan/compute flow:

* bind-time ``WiringError`` on incompatible definitions, before any work;
* totality — N questions ⇒ N records, in declaration order;
* Absent ``on``/aux input ⇒ NOT_APPLICABLE record preserving the cause
  (missing key is still a wiring error — design L2);
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
    serialize_trace,
)

from metrics.helpers import (
    PRODUCTION_EXECUTOR,
    code_test_trace,
    evaluation_procedure,
    external_trace,
    fake_executor_always,
    output_budget_run,
    procedure_trace,
    scripted_batch,
    wall_clock_run,
)


# ---------------------------------------------------------------------------
# Definition helpers (lazy metrics imports inside).
# ---------------------------------------------------------------------------


def _definition(questions) -> object:
    from dr_code.eval import MetricExtractionDefinition

    return MetricExtractionDefinition(
        definition_id="def", version="1", questions=tuple(questions)
    )


def _q(metric_name: str, on: str = "input", **settings) -> object:
    from dr_code.eval import MetricQuestionBinding
    from dr_code.metrics import MetricName

    return MetricQuestionBinding(
        metric=MetricName(metric_name), on=on, settings=settings
    )


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
    """ast_stats requires CODE; a TEXT key is a kind mismatch (bind-time)."""
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
    """compressed_length requires a valid compression config; bad settings wire."""
    with pytest.raises(ValueError):
        _definition(
            [
                _q(
                    "compressed_length",
                    compression={"method": "gzip", "level": 99},
                )
            ]
        )
    assert counting_executor.call_count == 0


def test_stale_resolved_operator_version_is_rejected_before_execution(
    counting_executor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics import MetricName, extract_metrics
    from dr_code.metrics.registry import REGISTRY

    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition([_q("text_stats")])
    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    operator_cls = REGISTRY[str(MetricName.TEXT_STATS)]
    monkeypatch.setattr(operator_cls, "VERSION", "new-live-version")

    with pytest.raises(ValueError, match="stale resolved operator versions"):
        extract_metrics(
            trace,
            metric_extraction=metric_extraction,
            evaluation_procedure=procedure,
            executor=counting_executor,
        )
    assert counting_executor.call_count == 0


def test_stale_resolved_operator_count_is_rejected_before_execution(
    counting_executor,
) -> None:
    from dr_code.metrics import extract_metrics

    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition([_q("text_stats")])
    metric_extraction = definition.materialize().model_copy(
        update={"resolved_operator_versions": ()}
    )
    procedure = evaluation_procedure(definition, metric_extraction)

    with pytest.raises(ValueError, match="stale resolved operator versions"):
        extract_metrics(
            trace,
            metric_extraction=metric_extraction,
            evaluation_procedure=procedure,
            executor=counting_executor,
        )
    assert counting_executor.call_count == 0


def test_missing_auxiliary_key_is_a_wiring_error(
    task, counting_executor
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
    records = _extract(definition, trace)
    assert len(records) == 3
    assert [r.question for r in records] == [
        "text_stats",
        "code_leakage",
        "ast_stats",
    ]


def test_same_metric_and_key_with_different_settings_have_distinct_identity() -> (
    None
):
    questions = (
        _q(
            "compressed_length",
            compression={"method": "gzip", "level": 1},
        ),
        _q(
            "compressed_length",
            compression={"method": "gzip", "level": 9},
        ),
    )
    definition = _definition(questions)
    records = _extract(
        definition,
        external_trace(
            {
                "input": TextArtifact(text="identity-sensitive"),
                "output": TextArtifact(text="identity-sensitive"),
            }
        ),
    )

    assert [record.question for record in records] == [
        "compressed_length",
        "compressed_length",
    ]
    concrete_questions = definition.materialize().questions
    assert [record.question_identity_hash for record in records] == [
        question.identity_hash() for question in concrete_questions
    ]
    assert len({record.question_identity_hash for record in records}) == 2
    assert all(
        fact.lineage.question_identity_hash == record.question_identity_hash
        for record in records
        for fact in record.facts
    )


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
        assert record.absence_mode.value == "preprocessing_failure"
        assert record.absence_cause == "extract: no code"
        assert record.fact_values() == {}


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
    assert record.absence_mode.value == "preprocessing_failure"
    assert record.absence_cause == "load: missing task"


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
    assert record.question == MetricName.TEXT_STATS


def test_operator_exception_with_empty_message_is_a_failure_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    operator_cls = REGISTRY[str(MetricName.TEXT_STATS)]

    def boom(self, value, aux, ctx):  # noqa: ANN001
        raise ValueError

    monkeypatch.setattr(operator_cls, "compute", boom)
    definition = _definition([_q("text_stats")])
    record = _extract(
        definition,
        external_trace(
            {
                "input": TextArtifact(text="hi"),
                "output": TextArtifact(text="hi"),
            }
        ),
    )[0]

    assert record.status.value == "operator_failure"
    assert record.failure_type == "ValueError"
    assert record.failure_message == ""
    assert (
        record.question_identity_hash
        == definition.questions[0].identity_hash()
    )


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
    assert record.question == MetricName.AST_STATS
    assert record.fact_values() == {}


# ===========================================================================
# Infrastructure errors raise; candidate terminations are data (L3).
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
    task, real_executor, monkeypatch: pytest.MonkeyPatch
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
        _extract(definition, trace, executor=real_executor)


def test_wall_clock_budget_is_candidate_data_not_infrastructure(task) -> None:
    """A wall-clock budget death is scored against the candidate as timeout
    cases, not a raised failure (batch_runner attribution parity)."""
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
    assert (
        record.fact_values()["timeout_count"]
        == record.fact_values()["total_cases"]
    )


def test_output_budget_is_candidate_data_not_infrastructure(
    task,
) -> None:
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
    assert (
        record.fact_values()["error_count"]
        == record.fact_values()["total_cases"]
    )
    assert record.fact_values()["timeout_count"] == 0


# ===========================================================================
# Two-phase execution + content-hash request dedupe (X-S4).
# ===========================================================================


def test_batch_dedupes_identical_code_test_executions(
    task, counting_executor
) -> None:
    """Identical submissions across a sweep execute once (at-most-once)."""
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


def test_batch_returns_one_record_tuple_per_trace(task, real_executor) -> None:
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=5.0)]
    )
    results = _extract_batch(
        definition, [trace, trace], executor=real_executor
    )
    assert isinstance(results, tuple)
    assert len(results) == 2
    for per_trace in results:
        assert isinstance(per_trace, tuple)
        assert len(per_trace) == 1


def test_prepopulated_execution_cache_skips_the_executor(
    task, counting_executor
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
    _extract_batch(definition, [trace, trace], executor=counting_executor)
    assert counting_executor.call_count == 0


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
            producer_id="pre",
            version="v1",
            definition_hash="a" * 64,
            preprocessing_config_hash="b" * 64,
            implementation_hash="c" * 64,
        ),
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )

    def answer(record):
        return (
            record.question,
            record.facts[0].lineage.operator_version,
            record.on_key,
            record.status,
            tuple(sorted(record.fact_values().items())),
        )

    assert [answer(r) for r in _extract(definition, external)] == [
        answer(r) for r in _extract(definition, preprocessing)
    ]


def test_code_test_record_values_exclude_timing(task, real_executor) -> None:
    """Determinism soft spot: timing stays out of record values so identical
    inputs reproduce (plan section 3)."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = code_test_trace(candidate, task)
    definition = _definition(
        [_q("code_test", on="input", timeout_seconds=2.0)]
    )
    record = _extract(definition, trace, executor=real_executor)[0]
    assert "elapsed_seconds" not in record.fact_values()


# ---------------------------------------------------------------------------
# Engine call wrappers (keep the lazy import in one place).
# ---------------------------------------------------------------------------


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    kwargs.setdefault("executor", PRODUCTION_EXECUTOR)
    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    return extract_metrics(
        procedure_trace(trace, procedure),
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        **kwargs,
    )


def _extract_batch(definition, traces, **kwargs):
    from dr_code.metrics import extract_metrics_batch

    kwargs.setdefault("executor", PRODUCTION_EXECUTOR)
    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    return extract_metrics_batch(
        tuple(procedure_trace(trace, procedure) for trace in traces),
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        **kwargs,
    )
