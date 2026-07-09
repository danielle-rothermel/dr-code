"""Transforms that operate on Python source as code.

Every source-level function here assumes its input is parseable Python and
raises `SyntaxError` when it is not. Tree-level helpers assume the caller
owns the tree. For best-effort transforms over text that only probably
contains code (raw LLM output, markdown, mixed prose), see
`dr_code.text_transforms` — its functions are total and never raise.
"""

from __future__ import annotations

import ast
from typing import Final

#: Prefix used for alpha-renamed local variables (`_v0`, `_v1`, ...).
RENAMED_LOCAL_PREFIX: Final[str] = "_v"

_SCOPE_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def _validate_python(source: str) -> None:
    """Raise `SyntaxError` if `source` is not parseable Python."""
    ast.parse(source)


def is_string_literal_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def strip_leading_docstring(
    node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> None:
    """Drop `node`'s leading docstring in place, keeping the body parseable."""
    body = node.body
    if body and is_string_literal_stmt(body[0]):
        node.body = body[1:] or [ast.Pass()]


def strip_docstrings_in_tree(tree: ast.AST) -> ast.AST:
    """Drop leading docstrings from every scope in `tree`, in place."""
    for node in ast.walk(tree):
        if isinstance(node, _SCOPE_NODE_TYPES):
            strip_leading_docstring(node)
    return tree


def strip_docstrings(source: str) -> str:
    """Remove all docstrings via an `ast.unparse` round-trip."""
    tree = strip_docstrings_in_tree(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def canonicalize(source: str) -> str:
    """Return the canonical form `equivalent` compares: docstrings stripped,
    `ast.unparse` round-trip."""
    return strip_docstrings(source)


def equivalent(a: str, b: str) -> bool:
    """True if `a` and `b` are equivalent under canonicalization.

    Total: unparseable input compares as not-equivalent instead of raising.
    """
    try:
        return canonicalize(a) == canonicalize(b)
    except SyntaxError:
        return False


class _AnnotationStripper(ast.NodeTransformer):
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


def strip_type_annotations(source: str) -> str:
    """Drop annotations: `def f(x: int) -> str:` -> `def f(x):`,
    `x: int = 1` -> `x = 1`."""
    tree = ast.parse(source)
    transformed = _AnnotationStripper().visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def _module_level_names(tree: ast.AST) -> set[str]:
    """Top-level names that must not be renamed."""
    names: set[str] = set()
    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            if isinstance(
                stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
            ):
                names.add(stmt.name)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
    return names


class _LocalRenamer(ast.NodeTransformer):
    """Per-function rewriting of locals to `_vN`, deterministic per AST."""

    def __init__(self, protected: set[str], *, rename_params: bool) -> None:
        self._protected = protected
        self._rename_params = rename_params

    def _function_args(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[ast.arg]:
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg:
            args.append(node.args.vararg)
        if node.args.kwarg:
            args.append(node.args.kwarg)
        return args

    def _rename_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        local_names: list[str] = []
        seen: set[str] = set()
        param_names = {arg.arg for arg in self._function_args(node)}
        skipped = set(self._protected)
        if not self._rename_params:
            skipped |= param_names

        def add(name: str) -> None:
            if name in seen or name in skipped:
                return
            seen.add(name)
            local_names.append(name)

        if self._rename_params:
            for arg in self._function_args(node):
                add(arg.arg)

        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        add(target.id)
            elif isinstance(sub, ast.AnnAssign | ast.AugAssign):
                target = sub.target
                if isinstance(target, ast.Name):
                    add(target.id)

        mapping = {
            name: f"{RENAMED_LOCAL_PREFIX}{i}"
            for i, name in enumerate(local_names)
        }
        if not mapping:
            return

        if self._rename_params:
            for arg in self._function_args(node):
                if arg.arg in mapping:
                    arg.arg = mapping[arg.arg]

        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in mapping:
                sub.id = mapping[sub.id]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self._rename_function(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self._rename_function(node)
        return self.generic_visit(node)


def alpha_rename_locals(source: str, *, rename_params: bool = True) -> str:
    """Alpha-rename function locals to `_v0`, `_v1`, ... per function.

    Module-level names (functions, classes, top-level assignments, imports)
    are preserved. With `rename_params=False`, parameters keep their names
    (they are part of the signature contract) and only body locals rename.
    """
    tree = ast.parse(source)
    protected = _module_level_names(tree)
    _LocalRenamer(protected, rename_params=rename_params).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _top_level_import_linenos(tree: ast.Module) -> set[int]:
    linenos: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            # `end_lineno` is 1-based and inclusive.
            for lineno in range(
                node.lineno, (node.end_lineno or node.lineno) + 1
            ):
                linenos.add(lineno)
    return linenos


def remove_top_level_imports(source: str) -> str:
    """Delete top-level import lines, keeping the rest of the source intact."""
    import_linenos = _top_level_import_linenos(ast.parse(source))
    if not import_linenos:
        return source
    return "".join(
        line
        for lineno, line in enumerate(
            source.splitlines(keepends=True), start=1
        )
        if lineno not in import_linenos
    )


def dedupe_imports(source: str) -> str:
    """Drop exact-duplicate import lines, keeping the first occurrence."""
    _validate_python(source)
    seen: set[str] = set()
    out: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        is_import = stripped.startswith(("import ", "from "))
        if is_import:
            key = line.rstrip()
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    if source.endswith("\n"):
        return "\n".join(out) + "\n"
    return "\n".join(out)


__all__ = [
    "RENAMED_LOCAL_PREFIX",
    "alpha_rename_locals",
    "canonicalize",
    "dedupe_imports",
    "equivalent",
    "is_string_literal_stmt",
    "remove_top_level_imports",
    "strip_docstrings",
    "strip_docstrings_in_tree",
    "strip_leading_docstring",
    "strip_type_annotations",
]
