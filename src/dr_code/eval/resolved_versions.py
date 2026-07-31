"""Resolve concrete metric-operator and preprocessing-step versions."""

from __future__ import annotations

from dr_code.implementation_identity import implementation_identity


class UnknownOperatorError(KeyError):
    """A metric name has no registered operator."""


class UnknownStepError(KeyError):
    """A step name has no registered implementation."""


def resolved_operator_version(metric_name: str) -> str:
    from dr_code.metrics.registry import REGISTRY

    operator = REGISTRY.get(metric_name)
    if operator is None:
        raise UnknownOperatorError(
            f"no operator registered for {metric_name!r}"
        )
    return str(operator.VERSION)


def resolved_operator_identity(metric_name: str) -> tuple[str, str]:
    from dr_code.metrics.registry import REGISTRY

    operator = REGISTRY.get(metric_name)
    if operator is None:
        raise UnknownOperatorError(
            f"no operator registered for {metric_name!r}"
        )
    return str(operator.VERSION), implementation_identity(operator)


def resolved_step_version(step_name: str) -> str:
    from dr_code.preprocessing.registry import REGISTRY

    step = REGISTRY.get(step_name)
    if step is None:
        raise UnknownStepError(f"no step registered for {step_name!r}")
    return str(step.VERSION)


def resolved_step_identity(step_name: str) -> tuple[str, str]:
    from dr_code.preprocessing.registry import REGISTRY

    step = REGISTRY.get(step_name)
    if step is None:
        raise UnknownStepError(f"no step registered for {step_name!r}")
    return str(step.VERSION), implementation_identity(step)


__all__ = [
    "UnknownOperatorError",
    "UnknownStepError",
    "implementation_identity",
    "resolved_operator_identity",
    "resolved_operator_version",
    "resolved_step_identity",
    "resolved_step_version",
]
