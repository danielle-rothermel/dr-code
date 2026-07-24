"""Canonical live-registry validation for executable metric configurations."""

from __future__ import annotations

import json
from collections.abc import Mapping

from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.registry import REGISTRY
from dr_code.eval.resolved_versions import implementation_identity


def validated_metric_operator(
    *,
    name: str,
    settings: Mapping[str, object],
    expected_version: str | None = None,
    expected_implementation: str | None = None,
) -> MetricOperator:
    """Resolve and validate one executable operator from the live registry."""

    operator_class = REGISTRY.get(name)
    if operator_class is None:
        raise ValueError(f"unknown metric operator {name!r}")
    live_version = str(operator_class.VERSION)
    if expected_version is not None and expected_version != live_version:
        raise ValueError(
            f"operator version mismatch for {name!r}: configured "
            f"{expected_version!r}, live version is {live_version!r}"
        )
    live_implementation = implementation_identity(operator_class)
    if (
        expected_implementation is not None
        and expected_implementation != live_implementation
    ):
        raise ValueError(
            f"operator implementation mismatch for {name!r}: configured "
            f"{expected_implementation!r}, live implementation is "
            f"{live_implementation!r}"
        )
    fact_units = operator_class.FACT_UNITS
    if not fact_units:
        raise ValueError(
            f"metric operator {name!r} must declare at least one fact"
        )
    if any(
        not isinstance(fact_name, str)
        or not fact_name
        or not isinstance(unit, str)
        or not unit
        for fact_name, unit in fact_units.items()
    ):
        raise ValueError(f"metric operator {name!r} has invalid FACT_UNITS")

    settings_dict = dict(settings)
    validated = operator_class.Settings.model_validate_json(
        json.dumps(settings_dict, allow_nan=False),
        strict=True,
    )
    return operator_class(validated)


__all__ = ["validated_metric_operator"]
