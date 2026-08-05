"""Metric-name vocabulary contracts."""

from __future__ import annotations


EXPECTED_METRIC_NAMES = {
    "text_stats",
    "code_leakage",
    "parse_outcome",
    "ast_stats",
    "compressed_length",
    "code_test",
}


def test_metric_name_is_a_strenum_of_the_six_families() -> None:
    from dr_code.metrics import MetricName

    assert {name.value for name in MetricName} == EXPECTED_METRIC_NAMES


def test_metric_name_members_round_trip_through_their_string_values() -> None:
    from dr_code.metrics import MetricName

    for value in EXPECTED_METRIC_NAMES:
        name = MetricName(value)
        assert name.value == value
        assert name == str(name)  # StrEnum serializes to plain JSON
