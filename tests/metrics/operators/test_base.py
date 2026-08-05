"""Metric-operator base contracts."""

from __future__ import annotations

import pytest


def test_a_result_field_without_a_declared_unit_fails_loudly() -> None:
    """A new fact cannot reach a record carrying an unlabelled value."""
    from dr_code.metrics.operators.base import OperatorResult

    class UndeclaredResult(OperatorResult):
        UNITS = {}

        widget_count: int

    with pytest.raises(ValueError, match="declares no unit"):
        UndeclaredResult(widget_count=1).to_facts()
