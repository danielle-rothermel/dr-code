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
from dataclasses import dataclass
from typing import Final

from dr_code.core.source.python_analysis import (
    AnnotationKind,
    AnnotationSite,
    _identifier_names,
    _lambda_locals,
    _scope_bindings,
    _scope_declarations,
    annotation_sites,
    function_locals,
    function_params,
    is_string_literal_stmt,
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
    _LexicalLocalRenamer(
        reserved=set(),
        rename_params=True,
        target=node,
        target_mapping=mapping,
    ).visit(node)
    return node


def _preserved_local_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Local bindings intentionally kept stable by the public transform.

    Nested definition names retain the transform's established output shape.
    A dotted import without `as` binds its top-level package; adding an alias
    would instead bind the imported leaf module and change attribute access.
    """
    names: set[str] = set()

    class PreservedNameVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            names.add(node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            names.add(node.name)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            names.add(node.name)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Import(self, node: ast.Import) -> None:
            names.update(
                alias.name.split(".")[0]
                for alias in node.names
                if alias.asname is None and "." in alias.name
            )

    visitor = PreservedNameVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return names


def _local_mapping(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    reserved: set[str],
    *,
    rename_params: bool,
) -> dict[str, str]:
    param_names = {arg.arg for arg in function_params(node)}
    preserved_names = _preserved_local_names(node)
    local_names = [
        name for name in function_locals(node) if name not in preserved_names
    ]
    if not rename_params:
        local_names = [name for name in local_names if name not in param_names]
    mapping: dict[str, str] = {}
    index = 0
    for name in local_names:
        candidate = f"{RENAMED_LOCAL_PREFIX}{index}"
        while candidate in reserved:
            index += 1
            candidate = f"{RENAMED_LOCAL_PREFIX}{index}"
        mapping[name] = candidate
        reserved.add(candidate)
        index += 1
    return mapping


@dataclass(frozen=True, slots=True)
class _ScopeFrame:
    kind: str
    bound_names: set[str]
    global_names: set[str]
    mapping: dict[str, str]


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, ast.List | ast.Tuple):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


class _LexicalLocalRenamer(ast.NodeTransformer):
    """Rename function-owned names without crossing lexical ownership."""

    def __init__(
        self,
        reserved: set[str],
        *,
        rename_params: bool,
        target: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
        target_mapping: dict[str, str] | None = None,
    ) -> None:
        self._reserved = reserved
        self._rename_params = rename_params
        self._target = target
        self._target_mapping = target_mapping
        self._scopes: list[_ScopeFrame] = []

    def _mapped_name(self, name: str, *, skip_current: bool = False) -> str:
        frames = self._scopes[:-1] if skip_current else self._scopes
        nested_scope_seen = skip_current
        for frame in reversed(frames):
            if frame.kind == "class" and nested_scope_seen:
                continue
            if name in frame.global_names:
                return name
            if name in frame.bound_names:
                return frame.mapping.get(name, name)
            nested_scope_seen = True
        return name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._mapped_name(node.id)
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:
        node.names = [
            self._mapped_name(name, skip_current=True) for name in node.names
        ]
        return node

    def visit_ExceptHandler(
        self, node: ast.ExceptHandler
    ) -> ast.ExceptHandler:
        if node.name is not None:
            node.name = self._mapped_name(node.name)
        self.generic_visit(node)
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".")[0]
            mapped_name = self._mapped_name(bound_name)
            if mapped_name != bound_name:
                alias.asname = mapped_name
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        for alias in node.names:
            bound_name = alias.asname or alias.name
            mapped_name = self._mapped_name(bound_name)
            if mapped_name != bound_name:
                alias.asname = mapped_name
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.MatchAs:
        if node.name is not None:
            node.name = self._mapped_name(node.name)
        self.generic_visit(node)
        return node

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.MatchStar:
        if node.name is not None:
            node.name = self._mapped_name(node.name)
        return node

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.MatchMapping:
        if node.rest is not None:
            node.rest = self._mapped_name(node.rest)
        self.generic_visit(node)
        return node

    def _function_mapping(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> dict[str, str]:
        if node is self._target:
            preserved_names = _preserved_local_names(node)
            return {
                name: replacement
                for name, replacement in (self._target_mapping or {}).items()
                if name in function_locals(node)
                and name not in preserved_names
            }
        if self._target is not None:
            return {}
        reserved = self._reserved | {
            replacement
            for frame in self._scopes
            for replacement in frame.mapping.values()
        }
        return _local_mapping(
            node,
            reserved,
            rename_params=self._rename_params,
        )

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
        global_names, _ = _scope_declarations(node.body)
        mapping = self._function_mapping(node)
        for argument in function_params(node):
            argument.arg = mapping.get(argument.arg, argument.arg)

        self._scopes.append(
            _ScopeFrame("function", bound_names, global_names, mapping)
        )
        node.body = [self.visit(statement) for statement in node.body]
        self._scopes.pop()
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._mapped_name(node.name)
        visited = self._visit_function(node)
        assert isinstance(visited, ast.FunctionDef)
        return visited

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        node.name = self._mapped_name(node.name)
        visited = self._visit_function(node)
        assert isinstance(visited, ast.AsyncFunctionDef)
        return visited

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.name = self._mapped_name(node.name)
        node.decorator_list = [
            self.visit(item) for item in node.decorator_list
        ]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [self.visit(item) for item in node.keywords]
        node.type_params = [
            self.visit(item) for item in getattr(node, "type_params", [])
        ]
        global_names, _ = _scope_declarations(node.body)
        self._scopes.append(
            _ScopeFrame(
                "class", set(_scope_bindings(node.body)), global_names, {}
            )
        )
        node.body = [self.visit(statement) for statement in node.body]
        self._scopes.pop()
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        node.args.defaults = [self.visit(item) for item in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in node.args.kw_defaults
        ]
        bound_names = _lambda_locals(node)
        self._scopes.append(_ScopeFrame("lambda", bound_names, set(), {}))
        node.body = self.visit(node.body)
        self._scopes.pop()
        return node

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        first, *remaining = node.generators
        first.iter = self.visit(first.iter)
        bound_names = set().union(
            *(_target_names(generator.target) for generator in node.generators)
        )
        self._scopes.append(
            _ScopeFrame("comprehension", bound_names, set(), {})
        )
        first.target = self.visit(first.target)
        first.ifs = [self.visit(item) for item in first.ifs]
        for generator in remaining:
            generator.iter = self.visit(generator.iter)
            generator.target = self.visit(generator.target)
            generator.ifs = [self.visit(item) for item in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
        self._scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        self._visit_comprehension(node)
        return node

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        self._visit_comprehension(node)
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        self._visit_comprehension(node)
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        self._visit_comprehension(node)
        return node


def alpha_rename_locals_in_tree(
    tree: ast.Module,
    *,
    rename_params: bool = True,
) -> ast.Module:
    """Alpha-rename function locals to `_v0`, `_v1`, ... in `tree`."""
    reserved = _identifier_names(tree)
    _LexicalLocalRenamer(
        reserved,
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
