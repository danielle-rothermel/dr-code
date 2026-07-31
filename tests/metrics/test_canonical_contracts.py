"""Metrics execute directly onto canonical eval facts and configs."""

from __future__ import annotations

import pytest

from dr_code.eval import (
    EvaluationProcedureDefinition,
    EvaluationTraceSource,
    MetricExtractionDefinition,
    MetricRecord,
    MetricQuestionBinding,
    PreprocessingDefinition,
    record_rows,
)
from dr_code.humaneval.batch_runner import PRODUCTION_EXECUTOR
from dr_code.metrics import extract_metrics
from dr_code.preprocessing import run_preprocessing
from dr_code.trace import (
    ExternalSource,
    SampleIdentity,
    TextArtifact,
    Trace,
    TraceProducer,
    WiringError,
    external_trace,
    sample_identity_hash,
)

_EXTERNAL_SOURCE = ExternalSource(
    source_id="canonical-contracts",
    content_digest="c" * 64,
)


def _trace(preprocessing_config_hash: str):
    text = TextArtifact(text="one two")
    trace = external_trace(
        {"input": text, "output": text},
        source=_EXTERNAL_SOURCE,
    )
    return Trace(
        values=trace.values,
        producer=TraceProducer(
            producer_id="pre",
            version="1",
            definition_hash="a" * 64,
            preprocessing_config_hash=preprocessing_config_hash,
            implementation_hash=PreprocessingDefinition(
                definition_id="pre",
                version="1",
                steps=(),
            )
            .materialize()
            .implementation_hash,
        ),
        step_facts=trace.step_facts,
    )


def _definition() -> MetricExtractionDefinition:
    return MetricExtractionDefinition(
        definition_id="metrics",
        version="1",
        questions=(MetricQuestionBinding(metric="text_stats", on="output"),),
    )


def _procedure(definition: MetricExtractionDefinition):
    preprocessing = PreprocessingDefinition(
        definition_id="pre",
        version="1",
        steps=(),
    ).materialize()
    return EvaluationProcedureDefinition(
        definition_id="procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=definition.materialize(),
    )


def test_engine_emits_unitized_facts_with_resolved_lineage() -> None:
    definition = _definition()
    metric_extraction = definition.materialize()
    procedure = _procedure(definition)
    trace = _trace(procedure.preprocessing_config_hash)
    record = extract_metrics(
        trace,
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        executor=PRODUCTION_EXECUTOR,
    )[0]
    assert record.question == "text_stats"
    assert record.evaluation_procedure_config_hash == (
        procedure.config_identity_hash
    )
    assert record.fact_values()["word_count"] == 2
    assert record.trace_producer == trace.producer
    assert record.operator.name == "text_stats"
    assert record.operator.version == "1"
    assert record.operator.settings == ()
    assert MetricRecord.model_validate_json(record.model_dump_json()) == record
    assert all(fact.unit for fact in record.facts)
    assert {
        (
            fact.lineage.operator,
            fact.lineage.operator_version,
            fact.lineage.evaluation_procedure_config_hash,
        )
        for fact in record.facts
    } == {("text_stats", "1", procedure.config_identity_hash)}


def test_sample_identity_survives_trace_record_and_flattening() -> None:
    sample_identity = SampleIdentity(
        sampling_config_identity="a" * 64,
        repeat_identity="b" * 64,
        ordinal=0,
        task_identity="task",
        identity_hash=sample_identity_hash(
            sampling_config_identity="a" * 64,
            repeat_identity="b" * 64,
            ordinal=0,
            task_identity="task",
        ),
    )
    definition = _definition()
    metric_extraction = definition.materialize()
    preprocessing = PreprocessingDefinition(
        definition_id="pre",
        version="1",
        steps=(),
    ).materialize()
    procedure = EvaluationProcedureDefinition(
        definition_id="procedure",
        version="1",
    ).materialize(
        preprocessing=preprocessing,
        metric_extraction=metric_extraction,
    )
    trace = run_preprocessing(
        preprocessing,
        TextArtifact(text="one two"),
        sample_identity=sample_identity,
    )

    record = extract_metrics(
        trace,
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        executor=PRODUCTION_EXECUTOR,
    )[0]

    assert trace.sample_identity == sample_identity
    assert record.sample_identity == sample_identity
    assert record_rows((record,))[0]["sample_identity"] == (
        sample_identity.model_dump(mode="python")
    )


def test_engine_rejects_a_mismatched_procedure() -> None:
    definition = _definition()
    other = MetricExtractionDefinition(
        definition_id="other",
        version="1",
        questions=(
            MetricQuestionBinding(metric="parse_outcome", on="output"),
        ),
    )
    with pytest.raises(WiringError, match="does not reference"):
        extract_metrics(
            _trace(_procedure(other).preprocessing_config_hash),
            metric_extraction=definition.materialize(),
            evaluation_procedure=_procedure(other),
            executor=PRODUCTION_EXECUTOR,
        )


def test_engine_rejects_mismatched_preprocessing_provenance() -> None:
    definition = _definition()
    metric_extraction = definition.materialize()
    procedure = _procedure(definition)
    with pytest.raises(WiringError, match="preprocessing config"):
        extract_metrics(
            _trace("f" * 64),
            metric_extraction=metric_extraction,
            evaluation_procedure=procedure,
            executor=PRODUCTION_EXECUTOR,
        )


def test_external_evaluation_preserves_external_producer() -> None:
    definition = _definition()
    metric_extraction = definition.materialize()
    procedure = EvaluationProcedureDefinition(
        definition_id="external-procedure",
        version="1",
    ).materialize_external(
        metric_extraction=metric_extraction,
        external_source=_EXTERNAL_SOURCE,
    )
    text = TextArtifact(text="one two")
    trace = external_trace(
        {"input": text, "output": text},
        source=_EXTERNAL_SOURCE,
    )

    record = extract_metrics(
        trace,
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        executor=PRODUCTION_EXECUTOR,
    )[0]

    assert procedure.trace_source is EvaluationTraceSource.EXTERNAL
    assert procedure.preprocessing_config_hash is None
    assert record.trace_producer == trace.producer
    assert record.trace_producer.producer_id == "external"


def test_distinct_external_sources_have_distinct_procedures_and_do_not_mix() -> (
    None
):
    metric_extraction = _definition().materialize()
    other_source = ExternalSource(
        source_id="other-source",
        content_digest="d" * 64,
    )
    first = EvaluationProcedureDefinition(
        definition_id="external-procedure",
        version="1",
    ).materialize_external(
        metric_extraction=metric_extraction,
        external_source=_EXTERNAL_SOURCE,
    )
    second = EvaluationProcedureDefinition(
        definition_id="external-procedure",
        version="1",
    ).materialize_external(
        metric_extraction=metric_extraction,
        external_source=other_source,
    )
    assert first.config_identity_hash != second.config_identity_hash

    text = TextArtifact(text="one two")
    wrong_trace = external_trace(
        {"input": text, "output": text},
        source=other_source,
    )
    with pytest.raises(WiringError, match="authenticated external source"):
        extract_metrics(
            wrong_trace,
            metric_extraction=metric_extraction,
            evaluation_procedure=first,
            executor=PRODUCTION_EXECUTOR,
        )


def test_trace_source_contracts_reject_the_other_producer_kind() -> None:
    definition = _definition()
    metric_extraction = definition.materialize()
    preprocessing_procedure = _procedure(definition)
    external_procedure = EvaluationProcedureDefinition(
        definition_id="external-procedure",
        version="1",
    ).materialize_external(
        metric_extraction=metric_extraction,
        external_source=_EXTERNAL_SOURCE,
    )
    text = TextArtifact(text="one two")
    external = external_trace(
        {"input": text, "output": text},
        source=_EXTERNAL_SOURCE,
    )

    with pytest.raises(WiringError, match="does not accept external"):
        extract_metrics(
            external,
            metric_extraction=metric_extraction,
            evaluation_procedure=preprocessing_procedure,
            executor=PRODUCTION_EXECUTOR,
        )
    with pytest.raises(WiringError, match="authenticated external source"):
        extract_metrics(
            _trace(preprocessing_procedure.preprocessing_config_hash),
            metric_extraction=metric_extraction,
            evaluation_procedure=external_procedure,
            executor=PRODUCTION_EXECUTOR,
        )
