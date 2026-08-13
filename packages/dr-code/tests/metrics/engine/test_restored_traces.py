from __future__ import annotations

import asyncio

from dr_code.trace import (
    CodeArtifact,
    ComponentCoordinate,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    StepCoordinate,
    TextArtifact,
    Trace,
    deserialize_trace,
    external_trace,
    serialize_trace,
)

CODE = "def add_one(x):\n    return x + 1\n"
TEXT = "some prose with def and return keywords\n"


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
    from dr_code.metrics import MeasuredRecord

    values = record.values if isinstance(record, MeasuredRecord) else ()
    return (
        record.identity.question.metric,
        record.identity.metric_version,
        record.identity.question.on_key,
        record.status,
        values,
    )


def _extract(definition, trace, **kwargs):
    from dr_code.metrics import extract_metrics

    return asyncio.run(extract_metrics(definition, trace, **kwargs))


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
        producer=_preprocessing_producer(),
    )
    definition = _mixed_definition()
    assert [_answer(r) for r in _extract(definition, external)] == [
        _answer(r) for r in _extract(definition, preprocessing)
    ]
