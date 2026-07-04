"""annotation_strip — drop type annotations from defs and assignments.

`def f(x: int) -> str:` becomes `def f(x):`. `x: int = 1` becomes `x = 1`.
"""

from __future__ import annotations

import ast
import time
from typing import ClassVar

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import DiagnosticSeverity, DiagnosticSource, NormalizerName
from code_eval.normalizers.base import Normalizer


class _Stripper(ast.NodeTransformer):
    def _strip_args(self, args: ast.arguments) -> None:
        for arg in (
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
        ):
            arg.annotation = None
        if args.vararg:
            args.vararg.annotation = None
        if args.kwarg:
            args.kwarg.annotation = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        self._strip_args(node.args)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.returns = None
        self._strip_args(node.args)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        # Drop pure annotations (no value); turn annotated assignments into
        # plain assignments.
        if node.value is None:
            return ast.Pass()
        return ast.Assign(
            targets=[node.target],
            value=node.value,
            type_comment=None,
        )


def annotation_strip(source: str) -> str:
    tree = ast.parse(source)
    transformed = _Stripper().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


class AnnotationStrip(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.ANNOTATION_STRIP

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        try:
            out = annotation_strip(source)
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
                        kind="annotation_strip_failed",
                        step=self.NAME.value,
                    ),
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=False,
            )
