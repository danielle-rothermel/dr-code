"""The settings base shared by metric declarations and operators.

Owned by its own module so a declaration can name an operator's settings
model without importing the operator package, which imports the record
models that nest declarations back.
"""

from __future__ import annotations

from dr_code.core.models import FrozenModel


class OperatorSettings(FrozenModel):
    """Validated parameters that determine an operator's semantics."""


__all__ = ["OperatorSettings"]
