"""Metric-fact unit vocabulary contracts."""

from __future__ import annotations


EXPECTED_FACT_UNITS = {
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


def test_metric_fact_unit_is_the_closed_unit_vocabulary() -> None:
    from dr_code.metrics import MetricFactUnit

    assert {unit.value for unit in MetricFactUnit} == EXPECTED_FACT_UNITS
