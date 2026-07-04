"""Remove existing `import` / `from ... import` statements."""

from __future__ import annotations

import ast
import random
from typing import ClassVar

from code_eval.models.corrupted_sample import CorruptedSample
from code_eval.names import InverseTransformName, RepairName
from code_eval.synthetic.inverse_transforms.base import InverseTransform


def _remove_imports(source: str) -> str:
    """Delete top-level import lines while keeping the rest of the source intact."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    import_linenos: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            # `end_lineno` is 1-based and inclusive.
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                import_linenos.add(ln)
    if not import_linenos:
        return source
    out_lines: list[str] = []
    for i, line in enumerate(source.splitlines(keepends=True), start=1):
        if i in import_linenos:
            continue
        out_lines.append(line)
    return "".join(out_lines)


class RemoveImports(InverseTransform):
    """Remove top-level import statements from the source."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.REMOVE_IMPORTS
    EXPECTED_RECOVERY_STEPS: ClassVar[frozenset[str]] = frozenset(
        {RepairName.IMPORT_RECOVERY.value}
    )

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_remove_imports(source),
            expected_recovery_steps=self.EXPECTED_RECOVERY_STEPS,
        )
