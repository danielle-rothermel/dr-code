"""Semantic equivalence via canonical AST round-trip.

Two programs are considered equivalent if, after both are reduced to a
canonical form (strip docstrings, `ast.unparse` round-trip), the resulting
sources are byte-identical.

This is intentionally a syntactic/structural equivalence — not a runtime
equivalence — so it can be applied to any parseable Python without execution.
For execution-based equivalence against HumanEvalPlus unit tests, see the
optional sandbox runner (out of scope for the validator itself).
"""

from __future__ import annotations

import ast


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the leading docstring from Module/FunctionDef/AsyncFunctionDef/ClassDef."""
    for node in ast.walk(tree):
        if isinstance(
            node,
            ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                # Replace docstring with a no-op `pass` if it was the only
                # body element, so the resulting AST still parses.
                if len(body) == 1:
                    node.body = [ast.Pass()]
                else:
                    node.body = body[1:]
    return tree


def canonicalize(source: str) -> str:
    """Return a canonical-form source string for `source`.

    Raises `SyntaxError` if the input does not parse.
    """
    tree = ast.parse(source)
    stripped = _strip_docstrings(tree)
    ast.fix_missing_locations(stripped)
    return ast.unparse(stripped)


def equivalent(a: str, b: str) -> bool:
    """True if `a` and `b` are semantically equivalent under canonicalization."""
    try:
        return canonicalize(a) == canonicalize(b)
    except SyntaxError:
        return False
