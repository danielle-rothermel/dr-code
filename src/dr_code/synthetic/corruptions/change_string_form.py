"""Swap f-string forms with .format() equivalents (for a few patterns)."""

from __future__ import annotations

import ast
import random
import re
from typing import ClassVar, cast

from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption

#: Match simple f-strings like f"hello {x}" or f'hello {x}' with one variable.
_FSTRING_RE = re.compile(r"""f(['"])([^"'{}]*)\{(\w+)\}([^"'{}]*)\1""")


def _f_to_format(source: str) -> str:
    """Rewrite simple f-strings as .format() calls."""

    def repl(m: re.Match[str]) -> str:
        pre, name, post = m.group(2), m.group(3), m.group(4)
        return f'"{pre}{{}}{post}".format({name})'

    return _FSTRING_RE.sub(repl, source)


def _is_docstring_stmt(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr):
        return False
    value = stmt.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _format_call_for_literal(value: str) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Constant(value="{}"),
            attr="format",
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=value)],
        keywords=[],
    )


class _FirstStringLiteralToFormat(ast.NodeTransformer):
    """Replace the first non-docstring string literal with a ``.format()`` call."""

    def __init__(self) -> None:
        self.changed = False

    def visit(self, node: ast.AST) -> ast.AST:
        if self.changed:
            return node
        return super().visit(node)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        new_body: list[ast.stmt] = []
        for idx, stmt in enumerate(node.body):
            if idx == 0 and _is_docstring_stmt(stmt):
                new_body.append(stmt)
            else:
                new_body.append(cast(ast.stmt, self.visit(stmt)))
        node.body = new_body
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        new_body: list[ast.stmt] = []
        for idx, stmt in enumerate(node.body):
            if idx == 0 and _is_docstring_stmt(stmt):
                new_body.append(stmt)
            else:
                new_body.append(cast(ast.stmt, self.visit(stmt)))
        node.body = new_body
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        new_body: list[ast.stmt] = []
        for idx, stmt in enumerate(node.body):
            if idx == 0 and _is_docstring_stmt(stmt):
                new_body.append(stmt)
            else:
                new_body.append(cast(ast.stmt, self.visit(stmt)))
        node.body = new_body
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        new_body: list[ast.stmt] = []
        for idx, stmt in enumerate(node.body):
            if idx == 0 and _is_docstring_stmt(stmt):
                new_body.append(stmt)
            else:
                new_body.append(cast(ast.stmt, self.visit(stmt)))
        node.body = new_body
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.changed or not isinstance(node.value, str):
            return node
        if not node.value or "\n" in node.value or len(node.value) > 80:
            return node
        self.changed = True
        return _format_call_for_literal(node.value)


class _WrapReturnWithFormatGuard(ast.NodeTransformer):
    """Wrap the first ``return`` in a tautological ``.format()`` guard."""

    def __init__(self) -> None:
        self.changed = False

    def visit(self, node: ast.AST) -> ast.AST:
        if self.changed:
            return node
        return super().visit(node)

    def visit_Return(self, node: ast.Return) -> ast.Return:
        if self.changed or node.value is None:
            return node
        self.changed = True
        format_empty = _format_call_for_literal("")
        guard = ast.Compare(
            left=format_empty,
            ops=[ast.Eq()],
            comparators=[ast.Constant(value="")],
        )
        node.value = ast.IfExp(test=guard, body=node.value, orelse=node.value)
        return node


def _literal_to_format_fallback(source: str) -> str:
    """Fallback when no f-strings exist: corrupt via ``.format()`` usage."""
    for converter_cls in (
        _FirstStringLiteralToFormat,
        _WrapReturnWithFormatGuard,
    ):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source
        converter = converter_cls()
        new_tree = converter.visit(tree)
        if converter.changed:
            ast.fix_missing_locations(new_tree)
            out = ast.unparse(new_tree)
            if source.endswith("\n") and not out.endswith("\n"):
                out += "\n"
            return out
    return source


class ChangeStringForm(Corruption):
    """Convert simple f-strings to `.format()` calls."""

    NAME: ClassVar[CorruptionName] = CorruptionName.CHANGE_STRING_FORM
    VERSION: ClassVar[str] = "0"

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        del (
            rng
        )  # deterministic fallback; rng reserved for future stochastic picks
        corrupted = _f_to_format(source)
        if corrupted == source:
            corrupted = _literal_to_format_fallback(source)
        return CorruptedSample(
            corrupted_source=corrupted,
        )
