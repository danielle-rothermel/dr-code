from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.trace import CodeArtifact, TextArtifact, external_trace

from ._helpers import _definition, _extract, _q


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
    with pytest.raises(ValidationError) as exc_info:
        _q(
            "compressed_length",
            compression={"method": "gzip", "level": 99},
        )

    assert [
        (error["type"], error["loc"]) for error in exc_info.value.errors()
    ] == [("value_error", ("compression", "gzip"))]


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
    record = _extract(definition, trace)[0]
    assert record.status.value == "operator_failure"
    assert record.failure.failure_type == "ValueError"
    assert record.failure.failure_message == "operator bug"
    assert record.identity.question.metric is MetricName.TEXT_STATS


def test_operator_result_violating_a_record_invariant_is_a_failure_record(
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

    class _NoFacts:
        def to_facts(self) -> tuple[()]:
            return ()

    def empty(self, value, aux, ctx):  # noqa: ANN001
        return _NoFacts()

    monkeypatch.setattr(operator_cls, "compute", empty)

    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )

    records = _extract(definition, trace)

    by_metric = {record.identity.question.metric: record for record in records}
    failed = by_metric[MetricName.TEXT_STATS]
    assert failed.status.value == "operator_failure"
    assert failed.failure.failure_type == "ValidationError"
    assert by_metric[MetricName.AST_STATS].status.value == "measured"


def test_planning_failure_is_isolated_from_unaffected_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    source = "def f(x):\n    return x\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=source),
            "output": CodeArtifact(source=source),
        }
    )

    operator_cls = REGISTRY[str(MetricName.TEXT_STATS)]

    def fail_planning(self, value, aux):  # noqa: ANN001
        raise ValueError("invalid execution plan")

    monkeypatch.setattr(operator_cls, "execution_requests", fail_planning)
    records = _extract(
        _definition(
            [_q("text_stats", on="input"), _q("ast_stats", on="input")]
        ),
        trace,
    )

    by_metric = {record.identity.question.metric: record for record in records}
    failed = by_metric[MetricName.TEXT_STATS]
    assert failed.status.value == "operator_failure"
    assert failed.failure.failure_type == "ValueError"
    assert by_metric[MetricName.AST_STATS].status.value == "measured"
