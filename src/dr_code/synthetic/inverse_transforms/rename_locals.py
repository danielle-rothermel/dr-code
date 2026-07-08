"""Alpha-rename local variable names to obscure names."""

from __future__ import annotations

import ast
import random
from typing import ClassVar

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import InverseTransformName
from dr_code.synthetic.inverse_transforms.base import InverseTransform


class _LocalRenamer(ast.NodeTransformer):
    """Rename Local-store names within function bodies to _vN."""

    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}
        self._counter = 0

    def _new_name(self) -> str:
        n = f"_v{self._counter}"
        self._counter += 1
        return n

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Compute renames for this function's locals.
        local_renames: dict[str, str] = {}
        # Parameters are not renamed (they're part of the signature contract).
        param_names = {a.arg for a in node.args.args}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id not in param_names
                    ):
                        local_renames.setdefault(target.id, self._new_name())
        # Apply renames using a focused walker.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in local_renames:
                sub.id = local_renames[sub.id]
        return node


def _rename(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    _LocalRenamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class RenameLocals(InverseTransform):
    """Rename local variables to opaque `_vN` form."""

    NAME: ClassVar[InverseTransformName] = InverseTransformName.RENAME_LOCALS

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        return CorruptedSample(
            corrupted_source=_rename(source),
        )
