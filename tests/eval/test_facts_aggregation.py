"""Facts, scores, compression references, and pure aggregation."""

from __future__ import annotations

import math

import pytest
from pydantic import StrictInt, ValidationError

from dr_code.eval import (
    AggregationDefinition,
    AggregationInput,
    AggregationOutput,
    AggregationStatus,
    Applicability,
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
    ConfigCoordinate,
    DefinitionRef,
    MetricFact,
    MetricQuestionTemplate,
    MetricExtractionTemplate,
    OperatorLineage,
    ReferenceResolutionError,
    SCORE_SCHEMA_VERSION,
    Score,
    aggregate,
    compression_ratio,
)
from dr_code.metrics import (
    MetricName,
    MetricRecord,
    extract_metrics,
    record_facts,
)
from dr_code.trace import TextArtifact, external_trace

_PROCEDURE = ConfigCoordinate(
    definition_ref=DefinitionRef(
        definition_id="procedure",
        version="1",
        schema_name="dr_code.evaluation_procedure.definition",
    )
)


def _lineage() -> OperatorLineage:
    return OperatorLineage(
        evaluation_procedure_config=_PROCEDURE,
        operator=MetricName.TEXT_STATS,
        operator_version="1",
        on_key="output",
    )


def _fact() -> MetricFact:
    return MetricFact(
        name="word_count",
        value=3,
        unit="word",
        applicability=Applicability.APPLICABLE,
        lineage=_lineage(),
    )


# ===========================================================================
# Metric facts: unit, applicability, lineage.
# ===========================================================================


def test_fact_preserves_unit_applicability_and_lineage() -> None:
    fact = _fact()
    assert fact.unit == "word"
    assert fact.applicability is Applicability.APPLICABLE
    assert fact.lineage.operator is MetricName.TEXT_STATS
    assert fact.lineage.operator_version == "1"
    assert MetricFact.model_validate_json(fact.model_dump_json()) == fact


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_metric_fact_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError, match="finite"):
        MetricFact(
            name="word_count",
            value=value,
            unit="word",
            applicability=Applicability.APPLICABLE,
            lineage=_lineage(),
        )


def test_applicable_fact_requires_a_value_and_no_reason() -> None:
    with pytest.raises(ValidationError, match="applicable metric facts"):
        MetricFact(
            name="word_count",
            value=None,
            unit="word",
            applicability=Applicability.APPLICABLE,
            lineage=_lineage(),
        )
    with pytest.raises(ValidationError, match="absence reason"):
        MetricFact(
            name="word_count",
            value=3,
            unit="word",
            applicability=Applicability.APPLICABLE,
            reason="unused",
            lineage=_lineage(),
        )


def test_not_applicable_fact_requires_a_reason_and_no_value() -> None:
    with pytest.raises(ValidationError, match="explicit reason"):
        MetricFact(
            name="average_word_length",
            value=None,
            unit="character_per_word",
            applicability=Applicability.NOT_APPLICABLE,
            lineage=_lineage(),
        )
    with pytest.raises(ValidationError, match="cannot carry a value"):
        MetricFact(
            name="average_word_length",
            value=1.0,
            unit="character_per_word",
            applicability=Applicability.NOT_APPLICABLE,
            reason="no words",
            lineage=_lineage(),
        )


def test_metric_fact_requires_an_explicit_unit() -> None:
    with pytest.raises(ValidationError, match="explicit unit"):
        MetricFact(
            name="word_count",
            value=3,
            unit="",
            applicability=Applicability.APPLICABLE,
            lineage=_lineage(),
        )


def test_procedure_config_coordinate_is_required_everywhere() -> None:
    with pytest.raises(ValidationError, match="evaluation_procedure_config"):
        OperatorLineage(
            operator=MetricName.TEXT_STATS,
            operator_version="1",
            on_key="output",
        )
    with pytest.raises(ValidationError, match="evaluation_procedure_config"):
        Score(
            schema_version=SCORE_SCHEMA_VERSION,
            name="quality",
            value=1.0,
            unit="ratio",
            derived_from=("text_stats.word_count",),
        )


def test_lineage_step_and_step_version_are_set_together() -> None:
    with pytest.raises(ValidationError, match="set together"):
        OperatorLineage(
            evaluation_procedure_config=_PROCEDURE,
            operator=MetricName.TEXT_STATS,
            operator_version="1",
            on_key="output",
            step="return_all",
        )


# ===========================================================================
# record_facts: a measured record projects onto declared, unitized facts.
# ===========================================================================


def _measured_record_and_procedure():
    from metrics.helpers import evaluation_procedure, procedure_trace

    template = MetricExtractionTemplate(
        definition_id="metrics",
        version="1",
        questions=(MetricQuestionTemplate(metric="text_stats", on="output"),),
    )
    metric_extraction = template.materialize()
    procedure = evaluation_procedure(metric_extraction)
    text = TextArtifact(text="one two")
    trace = procedure_trace(
        external_trace({"input": text, "output": text}), procedure
    )
    record = extract_metrics(
        metric_extraction.definition,
        trace,
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
    )[0]
    return record, procedure


def test_record_facts_cover_every_declared_fact_with_its_unit() -> None:
    from dr_code.metrics.registry import REGISTRY

    record, procedure = _measured_record_and_procedure()
    facts = record_facts(record, evaluation_procedure=procedure)
    units = REGISTRY[MetricName.TEXT_STATS.value].FACT_UNITS

    assert {fact.name for fact in facts} == set(units)
    assert {fact.name: fact.unit for fact in facts} == dict(units)
    assert all(
        fact.lineage.evaluation_procedure_config == procedure.coordinate()
        for fact in facts
    )
    assert all(fact.lineage.on_key == record.on_key for fact in facts)


def test_record_facts_report_an_undefined_fact_as_not_applicable() -> None:
    record, procedure = _measured_record_and_procedure()
    facts = record_facts(record, evaluation_procedure=procedure)
    average = next(
        fact for fact in facts if fact.name == "average_word_length"
    )

    assert record.values["average_word_length"] is not None
    assert average.applicability is Applicability.APPLICABLE

    empty_record = record.model_copy(
        update={"values": {**record.values, "average_word_length": None}}
    )
    undefined = next(
        fact
        for fact in record_facts(empty_record, evaluation_procedure=procedure)
        if fact.name == "average_word_length"
    )
    assert undefined.applicability is Applicability.NOT_APPLICABLE
    assert undefined.value is None
    assert undefined.reason


def test_record_facts_are_empty_for_non_measured_records() -> None:
    record, procedure = _measured_record_and_procedure()
    failed = record.model_copy(
        update={
            "status": "operator_failure",
            "values": {},
            "failure_type": "ValueError",
            "failure_message": "boom",
        }
    )
    assert record_facts(failed, evaluation_procedure=procedure) == ()


# ===========================================================================
# Historical artifacts stay loadable across live operator upgrades.
# ===========================================================================


def test_historical_facts_and_scores_round_trip_after_operator_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics.operators.text_stats import TextStats
    from dr_code.metrics.settings import OperatorSettings

    class UpgradedSettings(OperatorSettings):
        required_new_setting: StrictInt

    fact = _fact()
    score = Score(
        schema_version=SCORE_SCHEMA_VERSION,
        name="quality",
        value=1.0,
        unit="ratio",
        evaluation_procedure_config=_PROCEDURE,
        derived_from=("text_stats.word_count",),
    )
    fact_json = fact.model_dump_json()
    score_json = score.model_dump_json()

    monkeypatch.setattr(TextStats, "VERSION", "future-version")
    monkeypatch.setattr(TextStats, "Settings", UpgradedSettings)
    monkeypatch.setattr(
        TextStats, "FACT_UNITS", {"future_fact": "future_unit"}
    )

    assert MetricFact.model_validate_json(fact_json) == fact
    assert Score.model_validate_json(score_json) == score


def test_scores_require_an_explicit_schema_version_and_sources() -> None:
    payload = Score(
        schema_version=SCORE_SCHEMA_VERSION,
        name="quality",
        value=1.0,
        unit="ratio",
        evaluation_procedure_config=_PROCEDURE,
        derived_from=("text_stats.word_count",),
    ).model_dump(mode="json")
    del payload["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        Score.model_validate(payload)
    with pytest.raises(ValidationError, match="source fact names"):
        Score(
            schema_version=SCORE_SCHEMA_VERSION,
            name="quality",
            value=1.0,
            unit="ratio",
            evaluation_procedure_config=_PROCEDURE,
            derived_from=(),
        )


def test_engine_records_stay_loadable_and_name_their_question() -> None:
    record, _ = _measured_record_and_procedure()
    assert MetricRecord.model_validate_json(record.model_dump_json()) == record
    assert record.metric is MetricName.TEXT_STATS
    assert record.on_key == "output"


# ===========================================================================
# Pure aggregation.
# ===========================================================================


def test_aggregation_is_explicit_about_missing_and_zero_denominators() -> None:
    propagate = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": "mean"})
    missing = aggregate(
        propagate,
        (AggregationInput(1.0), AggregationInput(None)),
    )
    assert missing.status is AggregationStatus.MISSING_DATA

    skip = AggregationDefinition(definition_id="agg", version="1").materialize(
        {"reduction": "mean", "missing_data": "skip"}
    )
    empty = aggregate(skip, (AggregationInput(None),))
    assert empty.status is AggregationStatus.ZERO_DENOMINATOR
    assert empty.value is None

    fail = AggregationDefinition(definition_id="agg", version="1").materialize(
        {
            "reduction": "mean",
            "missing_data": "skip",
            "zero_denominator": "error",
        }
    )
    with pytest.raises(ZeroDivisionError, match="zero contributing"):
        aggregate(fail, (AggregationInput(None),))


def test_aggregation_is_pure_and_counts_inputs() -> None:
    config = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": "mean"})
    inputs = (
        AggregationInput(1.0),
        AggregationInput(3.0),
        AggregationInput(None, applicable=False),
    )
    assert aggregate(config, inputs) == aggregate(config, inputs)
    output = aggregate(config, inputs)
    assert output.value == 2.0
    assert (
        output.count_total,
        output.count_applicable,
        output.count_present,
    ) == (3, 2, 2)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_aggregation_represents_overflow_as_finite_json(
    reduction: str,
) -> None:
    config = AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize({"reduction": reduction})

    output = aggregate(
        config,
        (AggregationInput(1e308), AggregationInput(1e308)),
    )

    assert output.status is AggregationStatus.NON_FINITE
    assert output.value is None
    assert (
        AggregationOutput.model_validate_json(output.model_dump_json())
        == output
    )


def test_compression_reference_resolution_and_zero_denominator() -> None:
    key = CompressionReferenceKey(namespace="dataset", name="ground_truth")
    artifact = CompressionReferenceArtifact(content=b"abcd")
    resolver = CompressionReferenceResolver.from_mapping({key: artifact})
    assert resolver.resolve(key) is artifact
    assert (
        compression_ratio(
            numerator_bytes=2,
            reference=artifact,
        )
        == 0.5
    )
    assert (
        compression_ratio(
            numerator_bytes=2,
            reference=CompressionReferenceArtifact(content=b""),
        )
        is None
    )
    with pytest.raises(ReferenceResolutionError):
        resolver.resolve(
            CompressionReferenceKey(namespace="dataset", name="missing")
        )
