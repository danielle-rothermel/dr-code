"""Project-wide base model.

Every pydantic model in `code_eval` inherits from `FrozenModel` so the
freeze/strictness contract is uniform: instances are immutable, unknown
fields are rejected, and any attempted assignment is re-validated.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Base class for all code-eval models."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        # `arbitrary_types_allowed=False` is the default; we explicitly want it.
        arbitrary_types_allowed=False,
    )
