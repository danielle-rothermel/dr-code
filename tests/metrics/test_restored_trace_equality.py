"""Restored-trace record equality.

The determinism promise: *same canonical inputs + same metric
identity/settings ⇒ same record*. Restored (deserialized) traces start with
cold caches but must measure identically to fresh traces. These tests cover
that promise across the three producer origins and the cache-hit path.
"""

from __future__ import annotations

from dr_code.execution.subprocess import SubprocessCompletedProcess
from dr_code.trace import (
    CodeArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    deserialize_trace,
    external_trace,
    serialize_trace,
)

from metrics.helpers import code_test_trace

CODE = "def add_one(x):\n    return x + 1\n"
TEXT = "some prose with def and return keywords\n"


def _mixed_definition():
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition

    return MetricsDefinition(
        definition_id="eq",
        version="1",
        questions=(
            MetricQuestion(metric=MetricName.TEXT_STATS, on="text"),
            MetricQuestion(
                metric=MetricName.CODE_LEAKAGE,
                on="text",
                settings={"task_names": ["add_one"]},
            ),
            MetricQuestion(metric=MetricName.PARSE_OUTCOME, on="code"),
            MetricQuestion(metric=MetricName.AST_STATS, on="code"),
            MetricQuestion(
                metric=MetricName.COMPRESSED_LENGTH,
                on="code",
                settings={"compression": {"method": "gzip", "level": 9}},
            ),
        ),
    )


def _namespace():
    return {
        "input": TextArtifact(text=TEXT),
        "output": TextArtifact(text=TEXT),
        "text": TextArtifact(text=TEXT),
        "code": CodeArtifact(source=CODE),
    }


def _answer(record):
    """Comparable projection: identity, status, and ordered values.

    Producer lineage legitimately differs across origins, so equality of the
    measured answer is what the determinism promise guarantees.
    """
    return (
        record.metric,
        record.metric_version,
        record.on_key,
        record.status,
        tuple(sorted(record.values.items())),
    )


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return extract_metrics(definition, trace, **kwargs)


def _stub_runner(*, source, input_text, timeout_seconds):  # noqa: ANN001
    return SubprocessCompletedProcess(returncode=0, stdout="[]", stderr="")


# ===========================================================================
# Fresh, deserialized, and external traces produce equal records.
# ===========================================================================


def test_deserialized_trace_measures_identically_to_fresh() -> None:
    fresh = external_trace(_namespace())
    restored = deserialize_trace(serialize_trace(fresh))
    definition = _mixed_definition()
    assert [_answer(r) for r in _extract(definition, fresh)] == [
        _answer(r) for r in _extract(definition, restored)
    ]


def test_external_trace_measures_identically_to_preprocessing_producer() -> (
    None
):
    external = external_trace(_namespace())
    preprocessing = Trace(
        values=_namespace(),
        producer=TraceProducer(
            producer_id="pre", version="v9", definition_hash="deadbeef"
        ),
    )
    definition = _mixed_definition()
    assert [_answer(r) for r in _extract(definition, external)] == [
        _answer(r) for r in _extract(definition, preprocessing)
    ]


def test_serialization_round_trip_is_lossless_for_artifacts() -> None:
    """Restored artifacts compare value-equal to the originals."""
    fresh = external_trace(_namespace())
    restored = deserialize_trace(serialize_trace(fresh))
    assert dict(restored.values) == dict(fresh.values)


def test_fresh_restored_and_external_all_yield_equal_answers() -> None:
    fresh = external_trace(_namespace())
    restored = deserialize_trace(serialize_trace(fresh))
    external = external_trace(_namespace())
    definition = _mixed_definition()
    answers = [
        [_answer(r) for r in _extract(definition, trace)]
        for trace in (fresh, restored, external)
    ]
    assert answers[0] == answers[1] == answers[2]


# ===========================================================================
# Execution-backed records are equal for restored and fresh traces.
# ===========================================================================


def _code_test_definition():
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition

    return MetricsDefinition(
        definition_id="ct",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="input",
                settings={"timeout_seconds": 5.0},
            ),
        ),
    )


def test_code_test_record_equal_across_fresh_and_restored(task) -> None:
    fresh = code_test_trace(CODE, task)
    restored = deserialize_trace(serialize_trace(fresh))
    definition = _code_test_definition()
    assert _answer(
        _extract(definition, fresh, run_in_subprocess=_stub_runner)[0]
    ) == _answer(
        _extract(definition, restored, run_in_subprocess=_stub_runner)[0]
    )


def test_batch_over_identical_traces_yields_equal_record_sets(task) -> None:
    """A sweep of identical traces yields identical record sets per trace."""
    from dr_code.metrics import extract_metrics_batch

    fresh = code_test_trace(CODE, task)
    restored = deserialize_trace(serialize_trace(fresh))
    definition = _code_test_definition()
    record_sets = extract_metrics_batch(
        definition,
        [fresh, restored, fresh],
        run_in_subprocess=_stub_runner,
    )
    answers = [[_answer(r) for r in records] for records in record_sets]
    assert answers[0] == answers[1] == answers[2]
