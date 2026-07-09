"""Transforms that operate on Python source as code.

Every source-level function here assumes its input is parseable Python and
raises `SyntaxError` when it is not. Tree-level helpers assume the caller
owns the tree. For best-effort transforms over text that only probably
contains code (raw LLM output, markdown, mixed prose), see
`dr_code.text_transforms` — its functions are total and never raise.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from typing import Final

from dr_code.code_analysis import (
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
    """Apply `mapping` to parameters and name references in one function."""
    if not mapping:
        return node
    for arg in function_params(node):
        if arg.arg in mapping:
            arg.arg = mapping[arg.arg]
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in mapping:
            sub.id = mapping[sub.id]
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
    return {
        name: f"{RENAMED_LOCAL_PREFIX}{i}"
        for i, name in enumerate(local_names)
    }


def alpha_rename_locals_in_tree(
    tree: ast.Module,
    *,
    rename_params: bool = True,
) -> ast.Module:
    """Alpha-rename function locals to `_v0`, `_v1`, ... in `tree`."""
    protected = module_level_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            mapping = _local_mapping(
                node,
                protected,
                rename_params=rename_params,
            )
            rename_locals_in_function(node, mapping)
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
