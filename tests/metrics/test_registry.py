"""Tests for the metrics operator registry."""

from __future__ import annotations


def test_registry_covers_every_metric_name() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    for name in MetricName:
        assert str(name) in REGISTRY, f"{name} missing from REGISTRY"


def test_registry_has_no_stray_keys() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    enum_values = {str(name) for name in MetricName}
    for key in REGISTRY:
        assert key in enum_values, f"registry key {key!r} not in MetricName"
