"""Tests for the metrics operator registry.

Covers the registry's coverage of ``MetricName`` and the unit declarations of
every result model reachable from a registered operator. Both walk ``REGISTRY``
rather than a maintained list, so registering an operator is enough to put it
under test.
"""

from __future__ import annotations

from typing import get_type_hints


def _registered_result_classes() -> dict[str, type]:
    """Every result model reachable from a registered operator.

    Discovered from ``REGISTRY`` rather than listed, so a new operator or a new
    result variant is covered the moment it is registered. Each operator's
    ``compute`` return annotation names its result model, and result subclasses
    are collected too: ``CompressedLengthWithReferenceResult`` is a subclass of
    ``CompressedLengthResult``, not a union member, so an annotation-only walk
    would miss the variant that actually carries the extra fields.
    """
    from dr_code.metrics.operators.base import OperatorResult
    from dr_code.metrics.registry import REGISTRY

    discovered: dict[str, type] = {}

    def collect(result_class: type) -> None:
        if not isinstance(result_class, type) or not issubclass(
            result_class, OperatorResult
        ):
            raise AssertionError(
                f"{result_class!r} is not an OperatorResult class"
            )
        discovered[result_class.__name__] = result_class
        for subclass in result_class.__subclasses__():
            collect(subclass)

    for name, operator in REGISTRY.items():
        annotation = get_type_hints(operator.compute).get("return")
        assert annotation is not None, f"{name}.compute has no return type"
        collect(annotation)
    return discovered


def test_every_registered_result_declares_units_for_exactly_its_fields() -> (
    None
):
    """``UNITS`` matches the field set exactly, for every reachable result.

    ``to_facts`` only raises for a field it is asked to project, so a unit
    declared for a field that no longer exists, or a field on a variant that is
    never exercised, survives every runtime path. Comparing the two sets
    statically is what closes that hole.
    """
    from dr_code.metrics.units import MetricFactUnit

    result_classes = _registered_result_classes()
    assert result_classes, "no operator result classes were discovered"

    for name, result_class in result_classes.items():
        fields = set(result_class.model_fields)
        units = set(result_class.UNITS)
        assert units - fields == set(), f"{name} declares units for no field"
        assert fields - units == set(), f"{name} has fields without a unit"
        for field_name, unit in result_class.UNITS.items():
            assert isinstance(unit, MetricFactUnit), f"{name}.{field_name}"


def test_registered_result_discovery_reaches_the_subclass_variant() -> None:
    """The discovery walk is load-bearing, so pin what it must reach.

    ``CompressedLengthWithReferenceResult`` is returned only when a reference
    key is configured; it is reachable from the registry solely as a subclass.
    """
    from dr_code.metrics.operators.compressed_length import (
        CompressedLengthResult,
        CompressedLengthWithReferenceResult,
    )

    discovered = _registered_result_classes()

    assert CompressedLengthResult.__name__ in discovered
    assert CompressedLengthWithReferenceResult.__name__ in discovered


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
