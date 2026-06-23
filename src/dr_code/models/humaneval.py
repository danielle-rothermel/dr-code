"""HumanEval+ task model."""

from __future__ import annotations

import ast
from typing import Any

from pydantic import model_validator

from dr_code.models.base import FrozenModel


class HumanEvalPlusTask(FrozenModel):
    """One task from HumanEvalPlus."""

    task_id: str
    entry_point: str
    prompt: str
    canonical_solution: str
    test: str
    expected_arity: int

    @model_validator(mode="before")
    @classmethod
    def _derive_expected_arity(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "expected_arity" in data:
            return data
        return {
            **data,
            "expected_arity": expected_arity_from_prompt(str(data["prompt"])),
        }

    @property
    def full_source(self) -> str:
        """Return the full ground-truth program (prompt + solution body)."""
        return self.prompt + self.canonical_solution


def expected_arity_from_prompt(prompt: str) -> int:
    """Return the positional arity of the first function in a task prompt."""
    module = ast.parse(f"{prompt.rstrip()}\n    pass\n")
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            args = node.args
            return len(args.posonlyargs) + len(args.args)
    msg = "HumanEval+ prompt does not define a function."
    raise ValueError(msg)
