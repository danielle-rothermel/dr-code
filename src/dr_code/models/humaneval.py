"""HumanEval+ task model."""

from __future__ import annotations

from dr_code.models.base import FrozenModel


class HumanEvalPlusTask(FrozenModel):
    """One task from HumanEvalPlus."""

    task_id: str
    entry_point: str
    prompt: str
    canonical_solution: str
    test: str

    @property
    def full_source(self) -> str:
        """Return the full ground-truth program (prompt + solution body)."""
        return self.prompt + self.canonical_solution
