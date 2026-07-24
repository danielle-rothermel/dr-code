"""Resolve operator/step names to their concrete implementation versions.

Metric Extraction and Evaluation Procedure identities MUST include the
*resolved* operator and step versions, not just their names. Bumping an
operator's ``VERSION`` (a behavior change) therefore changes the Config
Identity Hash even when the declared question set is unchanged.
"""

from __future__ import annotations

from dr_code.metrics.registry import REGISTRY as METRIC_REGISTRY
from dr_code.preprocessing.registry import REGISTRY as STEP_REGISTRY


class UnknownOperatorError(KeyError):
    """A metric name has no registered operator."""


class UnknownStepError(KeyError):
    """A step name has no registered step."""


def resolved_operator_version(metric_name: str) -> str:
    """Return the registered operator's ``VERSION`` for ``metric_name``."""

    operator = METRIC_REGISTRY.get(metric_name)
    if operator is None:
        raise UnknownOperatorError(
            f"no operator registered for {metric_name!r}"
        )
    return str(operator.VERSION)


def resolved_step_version(step_name: str) -> str:
    """Return the registered step's ``VERSION`` for ``step_name``."""

    step = STEP_REGISTRY.get(step_name)
    if step is None:
        raise UnknownStepError(f"no step registered for {step_name!r}")
    return str(step.VERSION)


__all__ = [
    "UnknownOperatorError",
    "UnknownStepError",
    "resolved_operator_version",
    "resolved_step_version",
]
