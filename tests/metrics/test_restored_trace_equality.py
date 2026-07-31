"""Restored-trace record equality (plan section 3, deliverable 4; design L2/L3).

The determinism promise: *same canonical inputs + same metric
identity/settings ⇒ same record*. Restored (deserialized) traces start with
cold caches but must measure identically to fresh traces. This file focuses
that promise across the three producer origins (X-S2) and the cache-hit path.

``dr_code.metrics`` is imported lazily inside each test so the suite collects
cleanly against the missing package and fails hard (never skips) when absent.
"""

from __future__ import annotations

from dr_code.trace import (
    CodeArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    deserialize_trace,
    serialize_trace,
)

from metrics.helpers import (
    code_test_trace,
    evaluation_procedure,
    external_trace,
    fake_executor_always,
    full_pass_batch,
    procedure_trace,
)

CODE = "def add_one(x):\n    return x + 1\n"
TEXT = "some prose with def and return keywords\n"


def _mixed_definition():
    from dr_code.eval import (
        MetricQuestionBinding,
        MetricExtractionDefinition,
    )
    from dr_code.metrics import MetricName

    return MetricExtractionDefinition(
        definition_id="eq",
        version="1",
        questions=(
            MetricQuestionBinding(metric=MetricName.TEXT_STATS, on="text"),
            MetricQuestionBinding(
                metric=MetricName.CODE_LEAKAGE,
                on="text",
                settings={"task_names": ["add_one"]},
            ),
            MetricQuestionBinding(metric=MetricName.PARSE_OUTCOME, on="code"),
            MetricQuestionBinding(metric=MetricName.AST_STATS, on="code"),
            MetricQuestionBinding(
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
    """Comparable projection: identity + status + ordered values (X-S2).

    Producer lineage legitimately differs across origins, so equality of the
    measured answer is what the determinism promise guarantees.
    """
    return (
        record.question,
        record.facts[0].lineage.operator_version,
        record.on_key,
        record.status,
        tuple(sorted(record.fact_values().items())),
    )


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    kwargs.setdefault("executor", _pass_all_executor())
    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    return extract_metrics(
        procedure_trace(trace, procedure),
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        **kwargs,
    )


def _pass_all_executor():
    return fake_executor_always(
        lambda call: full_pass_batch(case_ids=call.request.item_ids)
    )


# ===========================================================================
# Fresh ≡ deserialized ≡ external (X-S2).
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
            producer_id="pre",
            version="v9",
            definition_hash="d" * 64,
            preprocessing_config_hash="c" * 64,
            implementation_hash="e" * 64,
        ),
    )
    definition = _mixed_definition()
    assert [_answer(r) for r in _extract(definition, external)] == [
        _answer(r) for r in _extract(definition, preprocessing)
    ]


def test_serialization_round_trip_is_lossless_for_artifacts() -> None:
    """Restored artifacts compare value-equal to the originals (design S3)."""
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
# Execution-backed records are deterministic too: restored ≡ fresh (X-S2).
# ===========================================================================


def _code_test_definition():
    from dr_code.eval import (
        MetricQuestionBinding,
        MetricExtractionDefinition,
    )
    from dr_code.metrics import MetricName

    return MetricExtractionDefinition(
        definition_id="ct",
        version="1",
        questions=(
            MetricQuestionBinding(
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
        _extract(definition, fresh, executor=_pass_all_executor())[0]
    ) == _answer(
        _extract(definition, restored, executor=_pass_all_executor())[0]
    )


def test_batch_over_identical_traces_yields_equal_record_sets(task) -> None:
    """A sweep of identical traces yields identical record sets per trace."""
    from dr_code.metrics import extract_metrics_batch

    fresh = code_test_trace(CODE, task)
    restored = deserialize_trace(serialize_trace(fresh))
    definition = _code_test_definition()
    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    record_sets = extract_metrics_batch(
        [
            procedure_trace(trace, procedure)
            for trace in (fresh, restored, fresh)
        ],
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        executor=_pass_all_executor(),
    )
    answers = [[_answer(r) for r in records] for records in record_sets]
    assert answers[0] == answers[1] == answers[2]
