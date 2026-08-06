from __future__ import annotations

from typing import get_type_hints


def _registered_result_classes() -> dict[str, type]:
    from dr_code.metrics.operators.base import OperatorResult
    from dr_code.metrics.registry import REGISTRY

    discovered: dict[str, type] = {}

    # Registered result annotations do not enumerate subclass variants.
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
    from dr_code.metrics.units import MetricFactUnit

    result_classes = _registered_result_classes()
    assert result_classes, "no operator result classes were discovered"

    for name, result_class in result_classes.items():
        # Static field/unit equality catches declarations runtime paths miss.
        fields = set(result_class.model_fields)
        units = set(result_class.UNITS)
        assert units - fields == set(), f"{name} declares units for no field"
        assert fields - units == set(), f"{name} has fields without a unit"
        for field_name, unit in result_class.UNITS.items():
            assert isinstance(unit, MetricFactUnit), f"{name}.{field_name}"


def test_registered_result_discovery_reaches_the_subclass_variant() -> None:
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
