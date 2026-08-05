"""Metric-record model contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ._builders import (
    _absent,
    _fact,
    _identity,
    _measured,
    _not_applicable,
    _operator_failure,
)


EXPECTED_RECORD_STATUSES = {
    "measured",
    "not_applicable",
    "operator_failure",
}


def test_record_status_is_the_three_way_answer_taxonomy() -> None:
    from dr_code.metrics import RecordStatus

    assert {
        status.value for status in RecordStatus
    } == EXPECTED_RECORD_STATUSES


def test_record_status_members_round_trip_through_their_string_values() -> (
    None
):
    from dr_code.metrics import RecordStatus

    for value in EXPECTED_RECORD_STATUSES:
        status = RecordStatus(value)
        assert status.value == value


def test_every_record_variant_carries_the_initial_schema_version() -> None:
    from dr_code.metrics import METRIC_RECORD_SCHEMA_VERSION

    assert METRIC_RECORD_SCHEMA_VERSION == 1
    for record in (_measured(), _not_applicable(), _operator_failure()):
        assert record.schema_version == METRIC_RECORD_SCHEMA_VERSION


def test_records_reject_any_other_schema_version() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MeasuredRecord

    with pytest.raises(ValidationError):
        MeasuredRecord(
            schema_version=2,  # type: ignore[arg-type]
            identity=_identity(),
            facts=(_fact(),),
        )


def test_record_union_is_discriminated_by_status() -> None:
    from dr_code.metrics import (
        METRIC_RECORD_ADAPTER,
        MeasuredRecord,
        NotApplicableRecord,
        OperatorFailureRecord,
    )

    for record, expected in (
        (_measured(), MeasuredRecord),
        (_not_applicable(), NotApplicableRecord),
        (_operator_failure(), OperatorFailureRecord),
    ):
        restored = METRIC_RECORD_ADAPTER.validate_json(
            record.model_dump_json()
        )
        assert type(restored) is expected
        assert restored == record


def test_record_variant_fields_are_the_documented_schema() -> None:
    from dr_code.metrics import (
        MeasuredRecord,
        NotApplicableRecord,
        OperatorFailureRecord,
    )

    shared = {"schema_version", "status", "identity"}
    assert set(MeasuredRecord.model_fields) == shared | {"facts"}
    assert set(NotApplicableRecord.model_fields) == shared | {"absence"}
    assert set(OperatorFailureRecord.model_fields) == shared | {"failure"}


def test_records_are_frozen() -> None:
    record = _measured()
    with pytest.raises(ValidationError) as exc_info:
        record.facts = record.facts  # type: ignore[misc]

    error = exc_info.value.errors()[0]
    assert error["type"] == "frozen_instance"
    assert error["loc"] == ("facts",)


def test_equal_records_compare_equal() -> None:
    """Records participate in structured comparison across runs."""
    assert _measured() == _measured()
    assert _not_applicable() == _not_applicable()
    assert _operator_failure() == _operator_failure()


def test_measured_and_not_applicable_records_are_never_equal() -> None:
    assert _measured() != _not_applicable()
    assert _not_applicable() != _operator_failure()


def test_metric_fact_accepts_every_scalar_type() -> None:
    from dr_code.metrics import MetricFactUnit

    facts = (
        _fact(name="int_val", value=42),
        _fact(name="float_val", value=3.14, unit=MetricFactUnit.RATIO),
        _fact(name="str_val", value="hello", unit=MetricFactUnit.IDENTIFIER),
        _fact(name="bool_val", value=True, unit=MetricFactUnit.BOOLEAN),
        _fact(name="none_val", value=None, unit=MetricFactUnit.TEXT),
    )
    record = _measured(facts=facts)
    assert [fact.value for fact in record.facts] == [
        42,
        3.14,
        "hello",
        True,
        None,
    ]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_metric_fact_rejects_non_finite_values(value: float) -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricFactUnit

    with pytest.raises(ValidationError):
        _fact(name="ratio", value=value, unit=MetricFactUnit.RATIO)


def test_metric_fact_requires_a_unit_from_the_closed_vocabulary() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricFact

    with pytest.raises(ValidationError):
        MetricFact(name="count", value=1, unit="furlongs")


@pytest.mark.parametrize("name", ["character_count.unit", "a.b", ".", "x."])
def test_metric_fact_rejects_a_dotted_name(name: str) -> None:
    # A fact named ``x.unit`` would occupy fact ``x``'s unit column in
    # ``record_rows``; banning the separator is what makes that scheme
    # collision-free.
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must not contain"):
        _fact(name=name, value=1)


def test_metric_fact_rejects_an_empty_name() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        _fact(name="", value=1)

    error = exc_info.value.errors(include_url=False)[0]
    assert error["type"] == "value_error"
    assert error["loc"] == ()
    assert str(error["ctx"]["error"]) == ("metric fact name must not be empty")


def test_facts_preserve_operator_declaration_order() -> None:
    facts = (
        _fact(name="second", value=2),
        _fact(name="first", value=1),
    )
    assert [fact.name for fact in _measured(facts=facts).facts] == [
        "second",
        "first",
    ]


def test_measured_records_require_at_least_one_fact() -> None:
    """An empty fact tuple is indistinguishable from the no-answer shape."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _measured(facts=())


def test_measured_records_reject_duplicate_fact_names() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _measured(facts=(_fact(name="count"), _fact(name="count")))


def test_not_applicable_record_nests_the_complete_absent() -> None:
    record = _not_applicable()
    assert record.absence == _absent()
    assert record.absence.failed_step == "extract"
    assert record.absence.failure_code == "no_candidates_extracted"
    assert record.absence.cause == "no code extracted"
    assert record.absence.propagated_through == ("clean",)


def test_operator_failure_record_nests_a_structured_failure() -> None:
    record = _operator_failure()
    assert record.failure.failure_type == "ValueError"
    assert record.failure.failure_message == "boom"


def test_operator_failure_requires_type_and_message() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import OperatorFailure

    with pytest.raises(ValidationError):
        OperatorFailure(failure_type="ValueError")
    with pytest.raises(ValidationError):
        OperatorFailure(failure_message="boom")
