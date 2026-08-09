from __future__ import annotations


EXPECTED_VALUE_UNITS = {
    "count",
    "ratio",
    "percent",
    "characters",
    "bytes",
    "lines",
    "depth",
    "boolean",
    "identifier",
    "text",
}


def test_metric_value_unit_is_the_closed_unit_vocabulary() -> None:
    from dr_code.metrics import MetricValueUnit

    assert {unit.value for unit in MetricValueUnit} == EXPECTED_VALUE_UNITS
