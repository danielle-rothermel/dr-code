"""Metrics execute under the eval kernel's procedure and producer contracts."""

from __future__ import annotations

import pytest

from dr_code.eval import (
    EvaluationProcedureDefinition,
    EvaluationTraceSource,
    MetricExtractionTemplate,
    MetricQuestionTemplate,
    PreprocessingTemplate,
)
from dr_code.metrics import MetricRecord, extract_metrics, record_facts
from dr_code.trace import (
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    TextArtifact,
    Trace,
    WiringError,
    external_trace,
)


def _preprocessing_coordinate() -> PreprocessingDefinitionCoordinate:
    return (
        PreprocessingTemplate(definition_id="pre", version="1", steps=())
        .materialize()
        .definition_coordinate()
    )


def _trace(definition: PreprocessingDefinitionCoordinate):
    text = TextArtifact(text="one two")
    trace = external_trace({"input": text, "output": text})
    return Trace(
        values=trace.values,
        producer=PreprocessingTraceProducer(definition=definition),
        step_facts=trace.step_facts,
    )


def _template() -> MetricExtractionTemplate:
    return MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(MetricQuestionTemplate(metric="text_stats", on="output"),),
    )


def _procedure(template: MetricExtractionTemplate):
    preprocessing = PreprocessingTemplate(
        definition_id="pre",
        version="1",
        steps=(),
    ).materialize()
    return EvaluationProcedureDefinition(
        definition_id="procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=template.materialize(),
    )


def test_engine_emits_unitized_facts_with_resolved_lineage() -> None:
    template = _template()
    metric_extraction = template.materialize()
    procedure = _procedure(template)
    trace = _trace(_preprocessing_coordinate())
    record = extract_metrics(
        metric_extraction.definition,
        trace,
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
    )[0]

    assert record.metric == "text_stats"
    assert record.metric_version == "0"
    assert record.on_key == "output"
    assert record.values["word_count"] == 2
    assert record.producer == trace.producer
    assert MetricRecord.model_validate_json(record.model_dump_json()) == record

    facts = record_facts(record, evaluation_procedure=procedure)
    assert facts
    assert all(fact.unit for fact in facts)
    assert {
        (fact.lineage.operator, fact.lineage.operator_version)
        for fact in facts
    } == {("text_stats", "0")}
    assert all(
        fact.lineage.evaluation_procedure_config == procedure.coordinate()
        for fact in facts
    )
    assert {fact.name for fact in facts} == set(record.values)


def test_engine_rejects_a_mismatched_procedure() -> None:
    template = _template()
    metric_extraction = template.materialize()
    other = MetricExtractionTemplate(
        definition_id="other",
        version="1",
        questions=(
            MetricQuestionTemplate(metric="parse_outcome", on="output"),
        ),
    )
    with pytest.raises(WiringError, match="does not reference"):
        extract_metrics(
            metric_extraction.definition,
            _trace(_preprocessing_coordinate()),
            metric_extraction=metric_extraction,
            evaluation_procedure=_procedure(other),
        )


def test_engine_rejects_a_definition_the_config_does_not_carry() -> None:
    template = _template()
    metric_extraction = template.materialize()
    procedure = _procedure(template)
    other = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(
            MetricQuestionTemplate(metric="parse_outcome", on="output"),
        ),
    ).materialize()
    with pytest.raises(WiringError, match="does not carry"):
        extract_metrics(
            other.definition,
            _trace(_preprocessing_coordinate()),
            metric_extraction=metric_extraction,
            evaluation_procedure=procedure,
        )


def test_engine_rejects_mismatched_preprocessing_provenance() -> None:
    template = _template()
    metric_extraction = template.materialize()
    procedure = _procedure(template)
    with pytest.raises(WiringError, match="preprocessing definition"):
        extract_metrics(
            metric_extraction.definition,
            _trace(
                PreprocessingDefinitionCoordinate(
                    definition_id="other-pre",
                    version="1",
                    steps=(),
                )
            ),
            metric_extraction=metric_extraction,
            evaluation_procedure=procedure,
        )


def test_external_evaluation_preserves_external_producer() -> None:
    template = _template()
    metric_extraction = template.materialize()
    procedure = EvaluationProcedureDefinition(
        definition_id="external-procedure",
        version="1",
    ).materialize_external(metric_extraction=metric_extraction)
    text = TextArtifact(text="one two")
    trace = external_trace({"input": text, "output": text})

    record = extract_metrics(
        metric_extraction.definition,
        trace,
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
    )[0]

    assert procedure.trace_source is EvaluationTraceSource.EXTERNAL
    assert procedure.preprocessing_config is None
    assert record.producer == trace.producer
    assert record.producer.kind == "external"


def test_trace_source_contracts_reject_the_other_producer_kind() -> None:
    template = _template()
    metric_extraction = template.materialize()
    preprocessing_procedure = _procedure(template)
    external_procedure = EvaluationProcedureDefinition(
        definition_id="external-procedure",
        version="1",
    ).materialize_external(metric_extraction=metric_extraction)
    text = TextArtifact(text="one two")
    external = external_trace({"input": text, "output": text})

    with pytest.raises(WiringError, match="does not accept external"):
        extract_metrics(
            metric_extraction.definition,
            external,
            metric_extraction=metric_extraction,
            evaluation_procedure=preprocessing_procedure,
        )
    with pytest.raises(WiringError, match="requires an external trace"):
        extract_metrics(
            metric_extraction.definition,
            _trace(_preprocessing_coordinate()),
            metric_extraction=metric_extraction,
            evaluation_procedure=external_procedure,
        )
