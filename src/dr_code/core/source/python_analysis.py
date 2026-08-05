from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class PythonSourceValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse_ok: bool
    parse_error: str | None
    compile_ok: bool
    compile_error: str | None


class AnnotationKind(StrEnum):
    PARAMETER = "parameter"
    RETURN = "return"
    VARIABLE = "variable"


class TextSiteKind(StrEnum):
    HASH_COMMENT = "hash_comment"
    DOCSTRING = "docstring"


@dataclass(frozen=True)
class SourceLocation:
    lineno: int
    col_offset: int


@dataclass(frozen=True)
class AnnotationSite:
    kind: AnnotationKind
    name: str | None
    annotation: ast.expr
    annotation_source: str
    location: SourceLocation
    owner: ast.FunctionDef | ast.AsyncFunctionDef | ast.AnnAssign
    has_value: bool


@dataclass(frozen=True)
class FunctionArgument:
    name: str
    annotation_source: str | None = None


@dataclass(frozen=True)
class FunctionSignatureSite:
    name: str
    signature_source: str
    arguments: tuple[FunctionArgument, ...]
    location: SourceLocation
    owner: ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class SourceTextSite:
    kind: TextSiteKind
    name: str | None
    text: str
    location: SourceLocation
    owner: ast.AST | None


def validate_python(source: str) -> None:
    ast.parse(source)


@dataclass(frozen=True)
class SourceValidationWithTree:
    validation: PythonSourceValidation
    tree: ast.Module | None


def validate_python_source_with_ast(
    source: str,
) -> SourceValidationWithTree:
    try:
        parsed_module = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        parse_ok = False
        parse_error = f"{type(exc).__name__}: {exc}"
        parsed_module = None
    else:
        parse_ok = True
        parse_error = None

    try:
        compile(
            parsed_module if parsed_module is not None else source,
            "<candidate>",
            "exec",
        )
    except (SyntaxError, ValueError) as exc:
        compile_ok = False
        compile_error = f"{type(exc).__name__}: {exc}"
    else:
        compile_ok = True
        compile_error = None

    return SourceValidationWithTree(
        validation=PythonSourceValidation(
            parse_ok=parse_ok,
            parse_error=parse_error,
            compile_ok=compile_ok,
            compile_error=compile_error,
        ),
        tree=parsed_module,
    )


def validate_python_source(source: str) -> PythonSourceValidation:
    return validate_python_source_with_ast(source).validation


def is_string_literal_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def equivalent(a: str, b: str) -> bool:
    try:
        # Local import avoids the python_analysis/python_transforms cycle.
        from dr_code.core.source.python_transforms import strip_docstrings

        return strip_docstrings(a) == strip_docstrings(b)
    except SyntaxError:
        return False


def module_level_names(tree: ast.Module) -> set[str]:
    return set(_scope_bindings(tree.body))


def top_level_import_linenos(tree: ast.Module) -> set[int]:
    linenos: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            for lineno in range(
                node.lineno, (node.end_lineno or node.lineno) + 1
            ):
                linenos.add(lineno)
    return linenos


def _location(node: ast.expr) -> SourceLocation:
    return SourceLocation(lineno=node.lineno, col_offset=node.col_offset)


def _annotation_site(
    *,
    kind: AnnotationKind,
    name: str | None,
    annotation: ast.expr,
    owner: ast.FunctionDef | ast.AsyncFunctionDef | ast.AnnAssign,
    has_value: bool,
) -> AnnotationSite:
    return AnnotationSite(
        kind=kind,
        name=name,
        annotation=annotation,
        annotation_source=ast.unparse(annotation),
        location=_location(annotation),
        owner=owner,
        has_value=has_value,
    )


def function_params(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
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


def find_function_node(
    tree: ast.AST,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if isinstance(tree, ast.FunctionDef | ast.AsyncFunctionDef):
        return tree
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ),
        None,
    )


def format_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({args}){returns}:"


def extract_function_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[FunctionArgument]:
    variadic_arg_ids = {
        id(arg)
        for arg in (node.args.vararg, node.args.kwarg)
        if arg is not None
    }
    return [
        FunctionArgument(
            name=arg.arg,
            annotation_source=ast.unparse(arg.annotation)
            if arg.annotation
            else None,
        )
        for arg in function_params(node)
        if id(arg) not in variadic_arg_ids
    ]


def _function_signature_site(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionSignatureSite:
    return FunctionSignatureSite(
        name=node.name,
        signature_source=format_function_signature(node),
        arguments=tuple(extract_function_args(node)),
        location=SourceLocation(
            lineno=node.lineno, col_offset=node.col_offset
        ),
        owner=node,
    )


def extract_function_signatures(tree: ast.AST) -> list[FunctionSignatureSite]:
    return [
        _function_signature_site(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def annotation_sites(tree: ast.AST) -> list[AnnotationSite]:
    sites: list[AnnotationSite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in function_params(node):
                if arg.annotation is None:
                    continue
                sites.append(
                    _annotation_site(
                        kind=AnnotationKind.PARAMETER,
                        name=arg.arg,
                        annotation=arg.annotation,
                        owner=node,
                        has_value=True,
                    )
                )
            if node.returns is not None:
                sites.append(
                    _annotation_site(
                        kind=AnnotationKind.RETURN,
                        name=None,
                        annotation=node.returns,
                        owner=node,
                        has_value=True,
                    )
                )
        elif isinstance(node, ast.AnnAssign):
            target_name = (
                node.target.id if isinstance(node.target, ast.Name) else None
            )
            sites.append(
                _annotation_site(
                    kind=AnnotationKind.VARIABLE,
                    name=target_name,
                    annotation=node.annotation,
                    owner=node,
                    has_value=node.value is not None,
                )
            )
    return sites


def _target_names(targets: Iterable[ast.expr]) -> Iterable[str]:
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id
        elif isinstance(target, ast.Starred):
            yield from _target_names((target.value,))
        elif isinstance(target, ast.List | ast.Tuple):
            yield from _target_names(target.elts)


class _ScopeDeclarationVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def visit_Global(self, node: ast.Global) -> None:
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocals.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


def _scope_declarations(body: Iterable[ast.stmt]) -> tuple[set[str], set[str]]:
    visitor = _ScopeDeclarationVisitor()
    for statement in body:
        visitor.visit(statement)
    return visitor.globals, visitor.nonlocals


class _ScopeBindingVisitor(ast.NodeVisitor):
    def __init__(self, excluded: set[str]) -> None:
        self._excluded = excluded
        self.names: list[str] = []
        self._seen: set[str] = set()

    def _add(self, name: str | None) -> None:
        if name is None or name in self._excluded or name in self._seen:
            return
        self._seen.add(name)
        self.names.append(name)

    def _add_targets(self, targets: Iterable[ast.expr]) -> None:
        for name in _target_names(targets):
            self._add(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._add_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._add(node.target.id)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._add_targets((node.target,))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._add_targets((item.optional_vars,))
                self.visit(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._add_targets((item.optional_vars,))
                self.visit(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add(node.name)
        self._visit_definition_expressions(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node.name)
        self._visit_definition_expressions(node)

    def _visit_definition_expressions(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(item for item in node.args.kw_defaults if item is not None),
            *(
                argument.annotation
                for argument in function_params(node)
                if argument.annotation is not None
            ),
            *(item for item in (node.returns,) if item is not None),
        ):
            self.visit(expression)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node.name)
        for expression in (
            *node.decorator_list,
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        ):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for expression in (
            *node.args.defaults,
            *(item for item in node.args.kw_defaults if item is not None),
        ):
            self.visit(expression)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self.generic_visit(node)
        self._add(node.name)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.generic_visit(node)
        self._add(node.rest)

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        self._add_targets((node.name,))
        self.generic_visit(node)


def _scope_bindings(
    body: Iterable[ast.stmt],
    *,
    initial: Iterable[str] = (),
) -> list[str]:
    globals_, nonlocals = _scope_declarations(body)
    visitor = _ScopeBindingVisitor(globals_ | nonlocals)
    for name in initial:
        visitor._add(name)
    for statement in body:
        visitor.visit(statement)
    return visitor.names


def function_locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return _scope_bindings(
        node.body,
        initial=(argument.arg for argument in function_params(node)),
    )


def _lambda_locals(node: ast.Lambda) -> set[str]:
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        *(item for item in (node.args.vararg, node.args.kwarg) if item),
    )
    visitor = _ScopeBindingVisitor(set())
    for argument in arguments:
        visitor._add(argument.arg)
    visitor.visit(node.body)
    return set(visitor.names)


def _type_parameter_names(node: ast.AST) -> set[str]:
    return {
        type_param.name
        for type_param in getattr(node, "type_params", ())
        if isinstance(
            type_param, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple
        )
    }


def _descendant_scope_binders(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    names: set[str] = set()
    for descendant in ast.walk(node):
        if descendant is node:
            continue
        if isinstance(descendant, ast.FunctionDef | ast.AsyncFunctionDef):
            names.update(function_locals(descendant))
            names.update(_type_parameter_names(descendant))
        elif isinstance(descendant, ast.ClassDef):
            names.update(_scope_bindings(descendant.body))
            names.update(_type_parameter_names(descendant))
            names.add("__class__")
        elif isinstance(descendant, ast.Lambda):
            names.update(_lambda_locals(descendant))
        elif isinstance(
            descendant,
            ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        ):
            for generator in descendant.generators:
                names.update(_target_names((generator.target,)))
        elif isinstance(descendant, ast.TypeAlias):
            names.update(_type_parameter_names(descendant))
    return names


def _identifier_names(tree: ast.AST) -> set[str]:
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
        and not node.args
        and not node.keywords
        for node in ast.walk(tree)
    ):
        names.add("__class__")
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            names.update(node.names)
        elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        elif isinstance(node, ast.TypeVar | ast.ParamSpec | ast.TypeVarTuple):
            names.add(node.name)
    return names


def extract_hash_comments(code_str: str) -> list[SourceTextSite]:
    tokens = tokenize.generate_tokens(io.StringIO(code_str).readline)
    return [
        SourceTextSite(
            kind=TextSiteKind.HASH_COMMENT,
            name=None,
            text=tok.string[1:].strip(),
            location=SourceLocation(
                lineno=tok.start[0],
                col_offset=tok.start[1],
            ),
            owner=None,
        )
        for tok in tokens
        if tok.type == tokenize.COMMENT
    ]


def _docstring_owner_name(node: ast.AST) -> str | None:
    return (
        node.name
        if isinstance(
            node,
            ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        )
        else None
    )


def extract_docstrings(tree: ast.AST) -> list[SourceTextSite]:
    docstrings: list[SourceTextSite] = []
    for node in ast.walk(tree):
        if not isinstance(
            node,
            ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        ):
            continue
        if not node.body or not is_string_literal_stmt(node.body[0]):
            continue
        string_node = node.body[0]
        if not (
            isinstance(string_node, ast.Expr)
            and isinstance(string_node.value, ast.Constant)
            and isinstance(string_node.value.value, str)
        ):
            continue
        docstrings.append(
            SourceTextSite(
                kind=TextSiteKind.DOCSTRING,
                name=_docstring_owner_name(node),
                text=string_node.value.value,
                location=SourceLocation(
                    lineno=string_node.lineno,
                    col_offset=string_node.col_offset,
                ),
                owner=node,
            )
        )
    return docstrings


def collect_comments(code_str: str, tree: ast.AST) -> str:
    items = [*extract_hash_comments(code_str), *extract_docstrings(tree)]
    return "\n".join(
        site.text
        for site in sorted(items, key=lambda item: item.location.lineno)
    )


__all__ = [
    "AnnotationKind",
    "AnnotationSite",
    "FunctionArgument",
    "FunctionSignatureSite",
    "PythonSourceValidation",
    "SourceValidationWithTree",
    "SourceLocation",
    "SourceTextSite",
    "TextSiteKind",
    "annotation_sites",
    "collect_comments",
    "equivalent",
    "extract_docstrings",
    "extract_function_args",
    "extract_function_signatures",
    "extract_hash_comments",
    "find_function_node",
    "format_function_signature",
    "function_locals",
    "function_params",
    "is_string_literal_stmt",
    "module_level_names",
    "top_level_import_linenos",
    "validate_python",
    "validate_python_source",
    "validate_python_source_with_ast",
]
