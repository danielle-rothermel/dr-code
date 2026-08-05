"""Frozen base model for metric-operator settings."""

from dr_code.base import FrozenModel


class OperatorSettings(FrozenModel):
    """Validated parameters that determine an operator's semantics."""


__all__ = ["OperatorSettings"]
