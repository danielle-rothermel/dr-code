"""Facts, records, compression references, and pure aggregation."""

from __future__ import annotations

import math

import pytest
from pydantic import StrictInt, ValidationError

from dr_code.eval import (
    AbsenceMode,
    AggregationDefinition,
    AggregationInput,
    AggregationOutput,
    AggregationStatus,
    Applicability,
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
    METRIC_RECORD_SCHEMA_VERSION,
    MetricFact,
    MetricRecord,
    OperatorCoordinates,
    OperatorLineage,
    ReferenceResolutionError,
    SCORE_SCHEMA_VERSION,
    Score,
    aggregate,
    compression_ratio,
    record_rows,
)
from dr_code.trace import TraceProducer

_PROCEDURE_HASH = "d" * 64


def _lineage() -> OperatorLineage:
    return OperatorLineage(
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        question_identity_hash=_question_identity(),
        operator="text_stats",
        operator_version="1",
        operator_implementation="c" * 64,
    )


def _producer() -> TraceProducer:
    return TraceProducer(
        producer_id="preprocessing",
        version="1",
        definition_hash="a" * 64,
        preprocessing_config_hash="b" * 64,
        implementation_hash="c" * 64,
    )


def _operator(name: str = "text_stats") -> OperatorCoordinates:
    return OperatorCoordinates(
        name=name,
        version="1",
        implementation_hash="c" * 64,
        settings=(),
    )


def _question_identity(
    name: str = "text_stats", *, on_key: str = "output"
) -> str:
    return _operator(name).question_identity_hash(on_key=on_key)


def _fact() -> MetricFact:
    return MetricFact(
        name="word_count",
        value=3,
        unit="word",
        applicability=Applicability.APPLICABLE,
        lineage=_lineage(),
    )


def _text_stats_facts() -> tuple[MetricFact, ...]:
    values: tuple[tuple[str, int | float | None, str], ...] = (
        ("character_count", 3, "character"),
        ("byte_count", 3, "byte"),
        ("line_count", 1, "line"),
        ("nonempty_line_count", 1, "line"),
        ("word_count", 3, "word"),
        ("average_word_length", None, "character_per_word"),
        ("punctuation_count", 0, "character"),
        ("symbol_count", 0, "character"),
    )
    return tuple(
        MetricFact(
            name=name,
            value=value,
            unit=unit,
            applicability=(
                Applicability.APPLICABLE
                if value is not None
                else Applicability.NOT_APPLICABLE
            ),
            reason=None if value is not None else "input contains no words",
            lineage=_lineage(),
        )
        for name, value, unit in values
    )


def test_fact_and_record_preserve_unit_applicability_and_lineage() -> None:
    record = MetricRecord.measured(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        facts=_text_stats_facts(),
    )
    assert record.fact_values()["word_count"] == 3
    word_count = next(
        fact for fact in record.facts if fact.name == "word_count"
    )
    assert word_count.unit == "word"
    assert word_count.applicability is Applicability.APPLICABLE
    assert word_count.lineage.operator_version == "1"
    row = record_rows((record,))[0]
    assert row["text_stats.word_count"] == 3
    assert "facts" not in row


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


@pytest.mark.parametrize("invalid_hash", ["short", "A" * 64, "g" * 64])
def test_procedure_hash_fields_require_lowercase_sha256(
    invalid_hash: str,
) -> None:
    with pytest.raises(ValidationError, match="evaluation_procedure"):
        OperatorLineage(
            evaluation_procedure_config_hash=invalid_hash,
            question_identity_hash=_question_identity(),
            operator="text_stats",
            operator_version="1",
            operator_implementation="c" * 64,
        )

    record_payload = MetricRecord.operator_failure(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        failure_type="ValueError",
        failure_message="failure",
    ).model_dump(mode="json")
    record_payload["evaluation_procedure_config_hash"] = invalid_hash
    with pytest.raises(ValidationError, match="evaluation_procedure"):
        MetricRecord.model_validate(record_payload)

    with pytest.raises(ValidationError, match="evaluation_procedure"):
        Score(
            schema_version=SCORE_SCHEMA_VERSION,
            name="quality",
            value=1.0,
            unit="ratio",
            evaluation_procedure_config_hash=invalid_hash,
            derived_from=("text_stats.word_count",),
        )


@pytest.mark.parametrize(
    "facts",
    [
        _text_stats_facts()[:-1],
        (
            *_text_stats_facts()[:-1],
            _text_stats_facts()[-1].model_copy(update={"name": "forged"}),
        ),
        (
            *_text_stats_facts()[:-1],
            _text_stats_facts()[-1].model_copy(update={"unit": "forged"}),
        ),
    ],
)
def test_measured_record_decodes_historical_fact_contract_structurally(
    facts: tuple[MetricFact, ...],
) -> None:
    record = MetricRecord.measured(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        facts=facts,
    )

    assert record.facts == facts


def test_measured_record_fact_contract_is_order_independent() -> None:
    record = MetricRecord.measured(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        facts=tuple(reversed(_text_stats_facts())),
    )

    assert {fact.name: fact.unit for fact in record.facts} == {
        fact.name: fact.unit for fact in _text_stats_facts()
    }


def test_record_authenticates_and_freezes_operator_settings() -> None:
    operator = OperatorCoordinates(
        name="compressed_length",
        version="1",
        implementation_hash="c" * 64,
        settings=(
            ("compression", {"method": "gzip", "level": 9}),
            ("reference_key", None),
        ),
    )
    nested = dict(operator.settings)["compression"]
    with pytest.raises(TypeError, match="do not support mutation"):
        nested["extra"] = True

    record = MetricRecord.operator_failure(
        question="compressed_length",
        question_identity_hash=operator.question_identity_hash(
            on_key="output"
        ),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=operator,
        failure_type="ValueError",
        failure_message="failure",
    )
    assert MetricRecord.model_validate_json(record.model_dump_json()) == record
    forged = record.model_dump(mode="json")
    forged["operator"]["settings"] = [
        ["compression", {"method": "gzip", "level": 8}],
        ["reference_key", None],
    ]
    with pytest.raises(ValueError, match="authenticate"):
        MetricRecord.model_validate(forged)


def test_operator_coordinates_are_structural_historical_data() -> None:
    historical = OperatorCoordinates(
        name="retired_operator",
        version="retired-version",
        implementation_hash="c" * 64,
        settings=(("retired_setting", True),),
    )
    assert (
        OperatorCoordinates.model_validate_json(historical.model_dump_json())
        == historical
    )
    with pytest.raises(ValueError, match="unique"):
        OperatorCoordinates(
            name="retired_operator",
            version="retired-version",
            implementation_hash="c" * 64,
            settings=(("duplicate", 1), ("duplicate", 2)),
        )
    with pytest.raises(ValueError, match="canonical key order"):
        OperatorCoordinates(
            name="retired_operator",
            version="retired-version",
            implementation_hash="c" * 64,
            settings=(("z", 1), ("a", 2)),
        )


def test_record_shapes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="at least one fact"):
        MetricRecord.measured(
            question="text_stats",
            question_identity_hash=_question_identity(),
            on_key="output",
            evaluation_procedure_config_hash=_PROCEDURE_HASH,
            trace_producer=_producer(),
            operator=_operator(),
            facts=(),
        )
    absent = MetricRecord.not_applicable(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        absence_mode=AbsenceMode.EMPTY_CANDIDATE_SET,
        cause="no candidates",
    )
    assert absent.facts == ()
    assert absent.absence_mode is AbsenceMode.EMPTY_CANDIDATE_SET


def test_operator_failure_accepts_present_empty_message() -> None:
    record = MetricRecord.operator_failure(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        failure_type="ValueError",
        failure_message="",
    )

    assert record.failure_message == ""
    with pytest.raises(ValueError, match="require failure type and message"):
        MetricRecord(
            schema_version=METRIC_RECORD_SCHEMA_VERSION,
            question="text_stats",
            question_identity_hash=_question_identity(),
            on_key="output",
            evaluation_procedure_config_hash=_PROCEDURE_HASH,
            trace_producer=_producer(),
            operator=_operator(),
            status="operator_failure",
            failure_type="ValueError",
            failure_message=None,
        )


def test_records_and_scores_require_explicit_schema_versions() -> None:
    record_payload = MetricRecord.operator_failure(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        failure_type="ValueError",
        failure_message="failure",
    ).model_dump(mode="json")
    score_payload = Score(
        schema_version=SCORE_SCHEMA_VERSION,
        name="quality",
        value=1.0,
        unit="ratio",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        derived_from=("text_stats.word_count",),
    ).model_dump(mode="json")

    del record_payload["schema_version"]
    del score_payload["schema_version"]
    with pytest.raises(ValidationError, match="schema_version"):
        MetricRecord.model_validate(record_payload)
    with pytest.raises(ValidationError, match="schema_version"):
        Score.model_validate(score_payload)


def test_historical_records_round_trip_after_live_operator_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics.operators.base import OperatorSettings
    from dr_code.metrics.operators.text_stats import TextStats

    class UpgradedSettings(OperatorSettings):
        required_new_setting: StrictInt

    record = MetricRecord.measured(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        facts=_text_stats_facts(),
    )
    score = Score(
        schema_version=SCORE_SCHEMA_VERSION,
        name="quality",
        value=1.0,
        unit="ratio",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        derived_from=("text_stats.word_count",),
    )
    record_json = record.model_dump_json()
    score_json = score.model_dump_json()

    monkeypatch.setattr(TextStats, "VERSION", "future-version")
    monkeypatch.setattr(TextStats, "Settings", UpgradedSettings)
    monkeypatch.setattr(
        TextStats, "FACT_UNITS", {"future_fact": "future_unit"}
    )

    assert MetricRecord.model_validate_json(record_json) == record
    assert Score.model_validate_json(score_json) == score


def test_measured_record_accepts_explicit_na_and_requires_lineage() -> None:
    record = MetricRecord.measured(
        question="text_stats",
        question_identity_hash=_question_identity(),
        on_key="output",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=_producer(),
        operator=_operator(),
        facts=_text_stats_facts(),
    )
    assert record.fact_values()["average_word_length"] is None
    with pytest.raises(ValueError, match="question identity"):
        MetricRecord.measured(
            question="text_stats",
            question_identity_hash="different-question",
            on_key="output",
            evaluation_procedure_config_hash=_PROCEDURE_HASH,
            trace_producer=_producer(),
            operator=_operator(),
            facts=(_fact(),),
        )
    with pytest.raises(ValueError, match="record question"):
        MetricRecord.measured(
            question="parse_outcome",
            question_identity_hash=_question_identity("parse_outcome"),
            on_key="output",
            evaluation_procedure_config_hash=_PROCEDURE_HASH,
            trace_producer=_producer(),
            operator=_operator("parse_outcome"),
            facts=(_fact(),),
        )


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
