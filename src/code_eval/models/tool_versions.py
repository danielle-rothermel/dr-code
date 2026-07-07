"""ToolVersions — snapshot of pinned/discovered tool versions."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import ToolName


class ToolVersions(FrozenModel):
    """Versions of every external tool the validator uses, captured at init."""

    python: str
    code_eval: str
    ruff: str
    ty: str | None = None

    def as_dict(self) -> dict[str, str]:
        """Return as a plain dict for embedding in `ValidationResult`."""
        d = {
            ToolName.PYTHON.value: self.python,
            ToolName.CODE_EVAL.value: self.code_eval,
            ToolName.RUFF.value: self.ruff,
        }
        if self.ty is not None:
            d[ToolName.TY.value] = self.ty
        return d
