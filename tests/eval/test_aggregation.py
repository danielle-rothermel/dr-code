"""Pure aggregation: determinism and explicit missing/zero-denominator."""

from __future__ import annotations

import pytest

from dr_code.eval.aggregation import (
    AggregationInput,
    AggregationStatus,
    aggregate,
)
from dr_code.eval.lifecycle import AggregationConfig, AggregationDefinition


def _config(**overrides: object) -> AggregationConfig:
    assignment: dict[str, object] = {"reduction": "mean"}
    assignment.update(overrides)
    return AggregationDefinition(
        definition_id="agg", version="1"
    ).materialize(assignment)  # type: ignore[arg-type]


def _values(*values: float) -> tuple[AggregationInput, ...]:
    return tuple(AggregationInput(value=value) for value in values)


def test_mean_is_pure_and_deterministic() -> None:
    config = _config()
    inputs = _values(1.0, 2.0, 3.0)
    first = aggregate(config, inputs)
    second = aggregate(config, inputs)
    assert first == second
    assert first.status is AggregationStatus.OK
    assert first.value == 2.0


def test_sum_reduction() -> None:
    config = _config(reduction="sum")
    output = aggregate(config, _values(1.0, 2.0, 4.0))
    assert output.status is AggregationStatus.OK
    assert output.value == 7.0


def test_missing_data_propagates_by_default() -> None:
    config = _config()
    inputs = (AggregationInput(value=1.0), AggregationInput(value=None))
    output = aggregate(config, inputs)
    assert output.status is AggregationStatus.MISSING_DATA
    assert output.value is None
    assert output.count_present == 1


def test_missing_data_skip_drops_missing() -> None:
    config = _config(missing_data="skip")
    inputs = (AggregationInput(value=2.0), AggregationInput(value=None))
    output = aggregate(config, inputs)
    assert output.status is AggregationStatus.OK
    assert output.value == 2.0


def test_all_not_applicable_yields_not_applicable() -> None:
    config = _config()
    inputs = (
        AggregationInput(value=None, applicable=False),
        AggregationInput(value=1.0, applicable=False),
    )
    output = aggregate(config, inputs)
    assert output.status is AggregationStatus.NOT_APPLICABLE
    assert output.value is None


def test_zero_denominator_is_explicit() -> None:
    config = _config(missing_data="skip")
    inputs = (AggregationInput(value=None), AggregationInput(value=None))
    output = aggregate(config, inputs)
    assert output.status is AggregationStatus.ZERO_DENOMINATOR
    assert output.value is None


def test_zero_denominator_error_policy_raises() -> None:
    config = _config(missing_data="skip", zero_denominator="error")
    inputs = (AggregationInput(value=None),)
    with pytest.raises(ZeroDivisionError):
        aggregate(config, inputs)


def test_output_is_provenance_free() -> None:
    output = aggregate(_config(), _values(1.0))
    # The output model carries only numeric/status/count fields, no
    # lifecycle, authority, objective, or reward fields.
    assert set(output.model_dump()) == {
        "status",
        "value",
        "count_total",
        "count_applicable",
        "count_present",
    }
