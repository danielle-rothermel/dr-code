from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    candidate,
    measured,
    not_applicable,
    operator_failure,
    policy,
    preprocessing_coordinate,
    question_coordinate,
    record_identity,
    sample_identity,
    slot,
)
from dr_code.evaluation import (
    AggregationEmptyDenominator,
    AggregationInput,
    AggregationMissing,
    AggregationNonFinite,
    AggregationNotApplicable,
    AggregationOk,
    AggregationSlot,
    AggregationStatistic,
    AggregationStatus,
    NotApplicablePolicy,
    aggregate,
)
from dr_code.metrics import MetricsDefinitionCoordinate, MetricValueUnit
from dr_code.trace import PreprocessingTraceProducer

EXPECTED_STATUSES = {
    "ok",
    "missing",
    "not_applicable",
    "empty_denominator",
    "non_finite",
}


def request_for(*slots: AggregationSlot, **overrides: object):
    return AggregationInput(
        policy=overrides.pop("policy", policy(**overrides)), slots=slots
    )


def test_status_vocabulary_is_closed() -> None:
    assert {member.value for member in AggregationStatus} == EXPECTED_STATUSES


def test_each_outcome_is_a_distinct_type_not_a_sentinel_float() -> None:
    assert "value" in AggregationOk.model_fields
    for result in (
        AggregationMissing,
        AggregationNotApplicable,
        AggregationEmptyDenominator,
        AggregationNonFinite,
    ):
        assert "value" not in result.model_fields


def test_missing_slot_produces_a_missing_result() -> None:
    result = aggregate(request_for(slot(None, ordinal=0)))
    assert isinstance(result, AggregationMissing)
    assert result.missing == (candidate(candidate_ordinal=0),)


def test_missing_names_every_empty_slot_in_input_order() -> None:
    result = aggregate(
        request_for(
            slot(None, ordinal=0),
            slot(measured(1), ordinal=1),
            slot(None, ordinal=2),
        )
    )
    assert isinstance(result, AggregationMissing)
    assert result.missing == (
        candidate(candidate_ordinal=0),
        candidate(candidate_ordinal=2),
    )


def test_missing_is_distinct_from_not_applicable() -> None:
    empty = aggregate(request_for(slot(None, ordinal=0)))
    refused = aggregate(
        request_for(
            slot(not_applicable(), ordinal=0),
            not_applicable=NotApplicablePolicy.FAIL,
        )
    )
    assert isinstance(empty, AggregationMissing)
    assert isinstance(refused, AggregationNotApplicable)
    assert type(empty) is not type(refused)


def test_missing_takes_precedence_over_a_refusing_record() -> None:
    result = aggregate(
        request_for(
            slot(None, ordinal=0),
            slot(not_applicable(), ordinal=1),
            not_applicable=NotApplicablePolicy.FAIL,
        )
    )
    assert isinstance(result, AggregationMissing)


def test_not_applicable_record_refuses_under_the_fail_policy() -> None:
    result = aggregate(
        request_for(
            slot(measured(1), ordinal=0),
            slot(not_applicable(), ordinal=1),
            not_applicable=NotApplicablePolicy.FAIL,
        )
    )
    assert isinstance(result, AggregationNotApplicable)
    assert result.refused == (candidate(candidate_ordinal=1),)


def test_operator_failure_refuses_under_the_default_policy() -> None:
    result = aggregate(
        request_for(
            slot(measured(1), ordinal=0),
            slot(operator_failure(), ordinal=1),
        )
    )
    assert isinstance(result, AggregationNotApplicable)
    assert result.refused == (candidate(candidate_ordinal=1),)


def test_not_applicable_and_operator_failure_are_separately_ruled() -> None:
    result = aggregate(
        request_for(
            slot(measured(4), ordinal=0),
            slot(not_applicable(), ordinal=1),
            slot(operator_failure(), ordinal=2),
            not_applicable=NotApplicablePolicy.ZERO,
            operator_failure=NotApplicablePolicy.EXCLUDE,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 2.0
    assert result.counted == 2
    assert result.excluded == 1


def test_every_slot_excluded_produces_an_empty_denominator() -> None:
    result = aggregate(
        request_for(
            slot(not_applicable(), ordinal=0),
            slot(not_applicable(), ordinal=1),
            not_applicable=NotApplicablePolicy.EXCLUDE,
        )
    )
    assert isinstance(result, AggregationEmptyDenominator)
    assert result.excluded == 2


def test_empty_denominator_is_distinct_from_a_zero_valued_mean() -> None:
    empty = aggregate(
        request_for(
            slot(not_applicable(), ordinal=0),
            not_applicable=NotApplicablePolicy.EXCLUDE,
        )
    )
    zero = aggregate(
        request_for(
            slot(not_applicable(), ordinal=0),
            not_applicable=NotApplicablePolicy.ZERO,
        )
    )
    assert isinstance(empty, AggregationEmptyDenominator)
    assert isinstance(zero, AggregationOk)
    assert zero.value == 0.0


def test_count_of_an_all_excluded_input_is_zero_not_empty() -> None:
    result = aggregate(
        request_for(
            slot(not_applicable(), ordinal=0),
            statistic=AggregationStatistic.COUNT,
            not_applicable=NotApplicablePolicy.EXCLUDE,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 0.0
    assert result.counted == 0
    assert result.excluded == 1


def test_overflowing_sum_produces_a_non_finite_result() -> None:
    huge = 1.0e308
    result = aggregate(
        request_for(
            slot(measured(huge), ordinal=0),
            slot(measured(huge), ordinal=1),
            statistic=AggregationStatistic.SUM,
        )
    )
    assert isinstance(result, AggregationNonFinite)
    assert result.counted == 2
    assert "sum" in result.reason


def test_overflow_is_reported_as_a_result_not_raised() -> None:
    huge = 1.0e308
    result = aggregate(
        request_for(
            *(slot(measured(huge), ordinal=at) for at in range(4)),
            statistic=AggregationStatistic.MEAN,
        )
    )
    assert isinstance(result, AggregationNonFinite)
    assert result.counted == 4
    assert "overflow" in result.reason


def test_an_int_value_too_large_for_a_float_is_a_non_finite_result() -> None:
    result = aggregate(
        request_for(
            slot(measured(10**309), ordinal=0),
            statistic=AggregationStatistic.SUM,
        )
    )
    assert isinstance(result, AggregationNonFinite)
    assert result.counted == 1


def test_non_finite_is_distinct_from_empty_denominator() -> None:
    huge = 1.0e308
    overflow = aggregate(
        request_for(
            slot(measured(huge), ordinal=0),
            slot(measured(huge), ordinal=1),
            statistic=AggregationStatistic.SUM,
        )
    )
    empty = aggregate(
        request_for(
            slot(not_applicable(), ordinal=0),
            not_applicable=NotApplicablePolicy.EXCLUDE,
        )
    )
    assert type(overflow) is not type(empty)


def test_an_ok_result_cannot_carry_a_non_finite_value() -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        AggregationOk(value=float("inf"), counted=1, excluded=0)


@pytest.mark.parametrize("field", ["counted", "excluded"])
def test_an_ok_result_rejects_a_negative_tally(field: str) -> None:
    payload = {"value": 1.0, "counted": 0, "excluded": 0}
    payload[field] = -1

    with pytest.raises(ValidationError) as exc_info:
        AggregationOk.model_validate(payload)

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "greater_than_equal"
    assert error["loc"] == (field,)


def test_a_non_finite_result_rejects_a_negative_count() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AggregationNonFinite(counted=-1, reason="sum overflowed")

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "greater_than_equal"
    assert error["loc"] == ("counted",)


def test_a_non_finite_result_requires_a_reason() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AggregationNonFinite(counted=1, reason="")

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "string_too_short"
    assert error["loc"] == ("reason",)


def test_mean_averages_the_counted_values() -> None:
    result = aggregate(
        request_for(
            slot(measured(1), ordinal=0),
            slot(measured(2), ordinal=1),
            slot(measured(6), ordinal=2),
            statistic=AggregationStatistic.MEAN,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 3.0
    assert result.counted == 3


def test_sum_totals_the_counted_values() -> None:
    result = aggregate(
        request_for(
            slot(measured(1), ordinal=0),
            slot(measured(2.5), ordinal=1),
            statistic=AggregationStatistic.SUM,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 3.5


def test_count_reports_how_many_slots_contributed() -> None:
    result = aggregate(
        request_for(
            slot(measured(9), ordinal=0),
            slot(measured(9), ordinal=1),
            statistic=AggregationStatistic.COUNT,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 2.0


def test_proportion_is_the_truthy_share_on_a_zero_to_one_scale() -> None:
    result = aggregate(
        request_for(
            slot(measured(1), ordinal=0),
            slot(measured(0), ordinal=1),
            slot(measured(1), ordinal=2),
            slot(measured(0), ordinal=3),
            statistic=AggregationStatistic.PROPORTION,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 0.5


def test_boolean_values_aggregate_as_one_and_zero() -> None:
    result = aggregate(
        request_for(
            slot(measured(True, unit=MetricValueUnit.BOOLEAN), ordinal=0),
            slot(measured(False, unit=MetricValueUnit.BOOLEAN), ordinal=1),
            statistic=AggregationStatistic.MEAN,
        )
    )
    assert isinstance(result, AggregationOk)
    assert result.value == 0.5


def test_distinct_candidate_coordinates_can_share_one_record_identity() -> (
    None
):
    result = aggregate(
        request_for(
            AggregationSlot(
                candidate=candidate(candidate_ordinal=0), record=measured(1)
            ),
            AggregationSlot(
                candidate=candidate(
                    sample=sample_identity(sample_id="sample-1"),
                    candidate_ordinal=0,
                ),
                record=measured(3),
            ),
        )
    )

    assert isinstance(result, AggregationOk)
    assert result.value == 2.0


def test_a_record_answering_another_question_raises() -> None:
    other = question_coordinate(on_key="somewhere_else")

    mismatched = measured(1, identity=record_identity(question=other))
    with pytest.raises(ValueError, match="but the policy aggregates"):
        aggregate(request_for(slot(mismatched, ordinal=0)))


@pytest.mark.parametrize(
    "rule",
    (
        NotApplicablePolicy.EXCLUDE,
        NotApplicablePolicy.ZERO,
        NotApplicablePolicy.FAIL,
    ),
)
@pytest.mark.parametrize("record_builder", (not_applicable, operator_failure))
def test_a_non_measured_record_answering_another_question_raises(
    rule: NotApplicablePolicy,
    record_builder,
) -> None:
    other = question_coordinate(on_key="somewhere_else")
    mismatched = record_builder(identity=record_identity(question=other))

    with pytest.raises(ValueError, match="but the policy aggregates"):
        aggregate(
            request_for(
                slot(mismatched, ordinal=0),
                not_applicable=rule,
                operator_failure=rule,
            )
        )


def test_a_record_without_the_policys_value_raises() -> None:
    with pytest.raises(ValueError, match="no value named"):
        aggregate(
            request_for(slot(measured(1, name="other_value"), ordinal=0))
        )


def test_records_with_different_metric_versions_raise() -> None:
    mismatched = measured(
        2, identity=record_identity(metric_version="another-version")
    )

    with pytest.raises(ValueError, match="incompatible metric versions"):
        aggregate(
            request_for(
                slot(measured(1), ordinal=0),
                slot(mismatched, ordinal=1),
            )
        )


def test_records_with_different_metrics_definitions_raise() -> None:
    question = question_coordinate()
    other_definition = MetricsDefinitionCoordinate(
        definition_id="other-metrics",
        version="0",
        questions=(question,),
    )
    mismatched = measured(
        2,
        identity=record_identity(metrics_definition=other_definition),
    )

    with pytest.raises(ValueError, match="incompatible metrics definitions"):
        aggregate(
            request_for(
                slot(measured(1), ordinal=0),
                slot(mismatched, ordinal=1),
            )
        )


def test_records_with_different_preprocessing_producers_raise() -> None:
    other_preprocessing = preprocessing_coordinate(version="another-version")
    other_producer = PreprocessingTraceProducer(definition=other_preprocessing)
    mismatched = measured(2, identity=record_identity(producer=other_producer))

    with pytest.raises(
        ValueError, match="incompatible preprocessing producers"
    ):
        aggregate(
            request_for(
                slot(measured(1), ordinal=0),
                AggregationSlot(
                    candidate=candidate(
                        preprocessing=other_preprocessing,
                        candidate_ordinal=1,
                    ),
                    record=mismatched,
                ),
            )
        )


def test_record_producer_must_match_its_slot_candidate_preprocessing() -> None:
    other_preprocessing = preprocessing_coordinate(version="another-version")
    mismatched = measured(
        identity=record_identity(
            producer=PreprocessingTraceProducer(definition=other_preprocessing)
        )
    )

    with pytest.raises(
        ValueError, match="does not match aggregation slot candidate"
    ):
        aggregate(request_for(slot(mismatched, ordinal=0)))


def test_values_with_the_same_name_but_different_units_raise() -> None:
    with pytest.raises(ValueError, match="incompatible units.*count.*bytes"):
        aggregate(
            request_for(
                slot(measured(1, unit=MetricValueUnit.COUNT), ordinal=0),
                slot(measured(2, unit=MetricValueUnit.BYTES), ordinal=1),
            )
        )


def test_a_non_numeric_metric_value_raises() -> None:
    text = measured("hello", unit=MetricValueUnit.TEXT)
    with pytest.raises(ValueError, match="non-numeric"):
        aggregate(request_for(slot(text, ordinal=0)))


@pytest.mark.parametrize(
    "statistic", (AggregationStatistic.SUM, AggregationStatistic.MEAN)
)
def test_opposite_overflowing_values_report_non_finite(
    statistic: AggregationStatistic,
) -> None:
    huge = 10**400
    result = aggregate(
        request_for(
            slot(measured(huge), ordinal=0),
            slot(measured(-huge), ordinal=1),
            statistic=statistic,
        )
    )

    assert isinstance(result, AggregationNonFinite)
    assert result.counted == 2


def test_input_requires_at_least_one_slot() -> None:
    with pytest.raises(ValidationError, match="at least one slot"):
        AggregationInput(policy=policy(), slots=())


def test_input_rejects_two_slots_at_the_same_coordinate() -> None:
    with pytest.raises(ValidationError, match="distinct candidate"):
        AggregationInput(
            policy=policy(),
            slots=(slot(measured(1), ordinal=0), slot(measured(2), ordinal=0)),
        )


def test_a_slot_defaults_to_empty_rather_than_to_a_record() -> None:
    assert AggregationSlot(candidate=candidate()).record is None


def test_aggregate_is_deterministic_over_repeated_calls() -> None:
    request = request_for(
        slot(measured(3), ordinal=0), slot(measured(4), ordinal=1)
    )
    assert aggregate(request) == aggregate(request)


def test_aggregate_does_not_mutate_its_input() -> None:
    request = request_for(
        slot(measured(3), ordinal=0), slot(not_applicable(), ordinal=1)
    )
    before = request.model_dump_json()
    aggregate(request)
    assert request.model_dump_json() == before


def test_aggregate_needs_no_registry() -> None:
    from types import MappingProxyType

    import dr_code.metrics.registry as registry_module

    request = request_for(slot(measured(5), ordinal=0))
    original = registry_module.REGISTRY
    try:
        registry_module.REGISTRY = MappingProxyType({})
        result = aggregate(request)
    finally:
        registry_module.REGISTRY = original
    assert isinstance(result, AggregationOk)
    assert result.value == 5.0


@pytest.mark.parametrize(
    "value",
    [
        AggregationOk(value=1.5, counted=2, excluded=1),
        AggregationMissing(missing=(candidate(),)),
        AggregationNotApplicable(refused=(candidate(),)),
        AggregationEmptyDenominator(excluded=3),
        AggregationNonFinite(
            counted=2, reason="sum of 2 values is not finite"
        ),
    ],
)
def test_result_round_trips_through_json(value) -> None:
    assert type(value).model_validate_json(value.model_dump_json()) == value


def test_input_round_trips_through_json() -> None:
    request = request_for(
        slot(measured(1), ordinal=0),
        slot(not_applicable(), ordinal=1),
        slot(None, ordinal=2),
    )
    assert AggregationInput.model_validate_json(request.model_dump_json()) == (
        request
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (AggregationMissing(missing=(candidate(),)), "missing"),
        (AggregationNotApplicable(refused=(candidate(),)), "not_applicable"),
        (AggregationEmptyDenominator(excluded=1), "empty_denominator"),
        (AggregationNonFinite(counted=1, reason="x"), "non_finite"),
        (AggregationOk(value=0.0, counted=1, excluded=0), "ok"),
    ],
)
def test_result_serializes_its_discriminating_status(
    result, expected: str
) -> None:
    assert result.model_dump(mode="json")["status"] == expected
