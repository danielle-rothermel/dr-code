"""name_normalize — alpha-rename local variables to `_v0`, `_v1`, ...

Module-level names (functions, classes, top-level assignments, imports)
are preserved. Function arguments and locals are renamed positionally per
function. The renaming is deterministic for a given AST.
"""

from __future__ import annotations

import ast
import time
from typing import ClassVar

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import DiagnosticSeverity, DiagnosticSource, NormalizerName
from code_eval.normalizers.base import Normalizer


def _collect_module_names(tree: ast.AST) -> set[str]:
    """Top-level names that must not be renamed."""
    names: set[str] = set()
    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(stmt.name)
            elif isinstance(stmt, ast.Import):
                for a in stmt.names:
                    names.add(a.asname or a.name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for a in stmt.names:
                    names.add(a.asname or a.name)
            elif isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
    return names


class _LocalRenamer(ast.NodeTransformer):
    """Per-function local rewriting."""

    def __init__(self, protected: set[str]) -> None:
        self._protected = protected

    def _rename_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Collect names introduced in this function: args + assignments.
        local_names: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name in seen or name in self._protected:
                return
            seen.add(name)
            local_names.append(name)

        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            add(arg.arg)
        if node.args.vararg:
            add(node.args.vararg.arg)
        if node.args.kwarg:
            add(node.args.kwarg.arg)

        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Name):
                        add(t.id)
            elif isinstance(sub, ast.AnnAssign | ast.AugAssign):
                tgt = sub.target
                if isinstance(tgt, ast.Name):
                    add(tgt.id)

        mapping = {name: f"_v{i}" for i, name in enumerate(local_names)}
        if not mapping:
            return

        # Rewrite args.
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if arg.arg in mapping:
                arg.arg = mapping[arg.arg]
        if node.args.vararg and node.args.vararg.arg in mapping:
            node.args.vararg.arg = mapping[node.args.vararg.arg]
        if node.args.kwarg and node.args.kwarg.arg in mapping:
            node.args.kwarg.arg = mapping[node.args.kwarg.arg]

        # Rewrite Name nodes in body.
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in mapping:
                sub.id = mapping[sub.id]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._rename_function(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._rename_function(node)
        return self.generic_visit(node)


def name_normalize(source: str) -> str:
    tree = ast.parse(source)
    protected = _collect_module_names(tree)
    _LocalRenamer(protected).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class NameNormalize(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.NAME_NORMALIZE

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        try:
            out = name_normalize(source)
            return NormalizedForm(
                normalizer=self.NAME,
                source=out,
                transformations_applied=(self.NAME.value,),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=True,
            )
        except SyntaxError as e:
            return NormalizedForm(
                normalizer=self.NAME,
                source=source,
                transformations_applied=(),
                diagnostics=(
                    Diagnostic(
                        source=DiagnosticSource.NORMALIZER,
                        severity=DiagnosticSeverity.WARNING,
                        message=str(e),
                        kind="name_normalize_failed",
                        step=self.NAME.value,
                    ),
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=False,
            )
