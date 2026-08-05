"""Transforms that operate on Python source as code.

Every source-level function here assumes its input is parseable Python and
raises `SyntaxError` when it is not. Tree-level helpers assume the caller
owns the tree. For best-effort transforms over text that only probably
contains code (raw LLM output, markdown, mixed prose), see
`dr_code.core.source.text_transforms` — its functions are total and never raise.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from typing import Final

from dr_code.core.source.python_analysis import (
    AnnotationKind,
    AnnotationSite,
    annotation_sites,
    function_locals,
    function_params,
    is_string_literal_stmt,
    module_level_names,
    top_level_import_linenos,
    validate_python,
)

#: Prefix used for alpha-renamed local variables (`_v0`, `_v1`, ...).
RENAMED_LOCAL_PREFIX: Final[str] = "_v"

SCOPE_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
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
        if isinstance(node, SCOPE_NODE_TYPES):
            strip_leading_docstring(node)
    return tree


def strip_docstrings(source: str) -> str:
    """Remove all docstrings via an `ast.unparse` round-trip."""
    tree = strip_docstrings_in_tree(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _should_strip(
    site: AnnotationSite,
    keep: Callable[[AnnotationSite], bool] | None,
) -> bool:
    return keep is None or not keep(site)


def _strip_function_annotations(
    site: AnnotationSite,
) -> None:
    if not isinstance(site.owner, ast.FunctionDef | ast.AsyncFunctionDef):
        return
    if site.kind is AnnotationKind.RETURN:
        site.owner.returns = None
        return
    for arg in function_params(site.owner):
        if arg.annotation is site.annotation:
            arg.annotation = None
            return


def _replacement_for_annassign(site: AnnotationSite) -> ast.stmt:
    owner = site.owner
    if not isinstance(owner, ast.AnnAssign):
        raise TypeError("annotation site owner is not an AnnAssign")
    if owner.value is None:
        return ast.copy_location(ast.Pass(), owner)
    return ast.copy_location(
        ast.Assign(
            targets=[owner.target], value=owner.value, type_comment=None
        ),
        owner,
    )


def _body_lists(tree: ast.AST) -> Iterable[list[ast.stmt]]:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list):
            yield body
        orelse = getattr(node, "orelse", None)
        if isinstance(orelse, list):
            yield orelse
        finalbody = getattr(node, "finalbody", None)
        if isinstance(finalbody, list):
            yield finalbody


def strip_type_annotations_in_tree(
    tree: ast.AST,
    *,
    keep: Callable[[AnnotationSite], bool] | None = None,
) -> ast.AST:
    """Drop selected annotations from `tree` in place."""
    sites = annotation_sites(tree)
    annassign_replacements = {
        id(site.owner): _replacement_for_annassign(site)
        for site in sites
        if site.kind is AnnotationKind.VARIABLE and _should_strip(site, keep)
    }
    for site in sites:
        if not _should_strip(site, keep):
            continue
        _strip_function_annotations(site)
    for body in _body_lists(tree):
        for index, stmt in enumerate(body):
            replacement = annassign_replacements.get(id(stmt))
            if replacement is not None:
                body[index] = replacement
    return tree


def strip_type_annotations(source: str) -> str:
    """Drop annotations: `def f(x: int) -> str:` -> `def f(x):`,
    `x: int = 1` -> `x = 1`."""
    tree = strip_type_annotations_in_tree(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def rename_locals_in_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    mapping: dict[str, str],
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Apply `mapping` within one function's lexical scope."""
    if not mapping:
        return node
    for arg in function_params(node):
        if arg.arg in mapping:
            arg.arg = mapping[arg.arg]

    class LocalNameRenamer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            return node

        def visit_AsyncFunctionDef(
            self, node: ast.AsyncFunctionDef
        ) -> ast.AsyncFunctionDef:
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
            return node

    renamer = LocalNameRenamer()
    for statement in node.body:
        renamer.visit(statement)
    return node


def _local_mapping(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    protected: set[str],
    *,
    rename_params: bool,
) -> dict[str, str]:
    param_names = {arg.arg for arg in function_params(node)}
    skipped = set(protected)
    if not rename_params:
        skipped |= param_names
    local_names = [
        name for name in function_locals(node) if name not in skipped
    ]
    if not rename_params:
        local_names = [name for name in local_names if name not in param_names]
    mapping: dict[str, str] = {}
    index = 0
    for name in local_names:
        candidate = f"{RENAMED_LOCAL_PREFIX}{index}"
        while candidate in protected:
            index += 1
            candidate = f"{RENAMED_LOCAL_PREFIX}{index}"
        mapping[name] = candidate
        protected.add(candidate)
        index += 1
    return mapping


class _LexicalLocalRenamer(ast.NodeTransformer):
    """Rename function-owned names without crossing lexical ownership."""

    def __init__(self, protected: set[str], *, rename_params: bool) -> None:
        self._module_names = protected
        self._rename_params = rename_params
        self._scopes: list[tuple[set[str], dict[str, str]]] = []

    def visit_Name(self, node: ast.Name) -> ast.Name:
        for bound_names, mapping in reversed(self._scopes):
            if node.id in bound_names:
                node.id = mapping.get(node.id, node.id)
                break
        return node

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        node.decorator_list = [
            self.visit(item) for item in node.decorator_list
        ]
        node.returns = self.visit(node.returns) if node.returns else None
        node.type_params = [
            self.visit(item) for item in getattr(node, "type_params", [])
        ]
        for argument in function_params(node):
            if argument.annotation is not None:
                argument.annotation = self.visit(argument.annotation)
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]

        bound_names = set(function_locals(node))
        protected = self._module_names | {
            renamed
            for _, active_mapping in self._scopes
            for renamed in active_mapping.values()
        }
        mapping = _local_mapping(
            node,
            set(protected),
            rename_params=self._rename_params,
        )
        for argument in function_params(node):
            argument.arg = mapping.get(argument.arg, argument.arg)

        self._scopes.append((bound_names, mapping))
        node.body = [self.visit(statement) for statement in node.body]
        self._scopes.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        visited = self._visit_function(node)
        assert isinstance(visited, ast.FunctionDef)
        return visited

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        visited = self._visit_function(node)
        assert isinstance(visited, ast.AsyncFunctionDef)
        return visited

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]
        bound_names = {
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg:
            bound_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            bound_names.add(node.args.kwarg.arg)
        self._scopes.append((bound_names, {}))
        node.body = self.visit(node.body)
        self._scopes.pop()
        return node


def alpha_rename_locals_in_tree(
    tree: ast.Module,
    *,
    rename_params: bool = True,
) -> ast.Module:
    """Alpha-rename function locals to `_v0`, `_v1`, ... in `tree`."""
    protected = module_level_names(tree)
    _LexicalLocalRenamer(
        protected,
        rename_params=rename_params,
    ).visit(tree)
    return tree


def alpha_rename_locals(source: str, *, rename_params: bool = True) -> str:
    """Alpha-rename function locals to `_v0`, `_v1`, ... per function.

    Module-level names (functions, classes, top-level assignments, imports)
    are preserved. With `rename_params=False`, parameters keep their names
    (they are part of the signature contract) and only body locals rename.
    """
    tree = ast.parse(source)
    alpha_rename_locals_in_tree(tree, rename_params=rename_params)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def remove_top_level_imports(source: str) -> str:
    """Delete top-level import lines, keeping the rest of the source intact."""
    import_linenos = top_level_import_linenos(ast.parse(source))
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
    validate_python(source)
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
    "SCOPE_NODE_TYPES",
    "alpha_rename_locals_in_tree",
    "alpha_rename_locals",
    "dedupe_imports",
    "remove_top_level_imports",
    "rename_locals_in_function",
    "strip_docstrings",
    "strip_docstrings_in_tree",
    "strip_leading_docstring",
    "strip_type_annotations",
    "strip_type_annotations_in_tree",
]
