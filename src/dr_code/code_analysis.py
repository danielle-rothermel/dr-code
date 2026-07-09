"""Analysis helpers for Python source treated as code.

Every source-level function here assumes its input is parseable Python and
raises `SyntaxError` when it is not, except documented-total diagnostics such
as `equivalent`. For transforms that modify parseable Python, see
`dr_code.code_transforms`; for total best-effort work over text that only
probably contains code, see `dr_code.text_transforms` and
`dr_code.text_analysis`.
"""

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
    """Raise `SyntaxError` if `source` is not parseable Python."""
    ast.parse(source)


@dataclass(frozen=True)
class SourceValidationWithTree:
    validation: PythonSourceValidation
    tree: ast.Module | None


def validate_python_source_with_ast(
    source: str,
) -> SourceValidationWithTree:
    """Return parse/compile diagnostics without raising, plus the parsed
    module for reuse (None when parsing failed)."""
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
    """Return parse and compile diagnostics without raising."""
    return validate_python_source_with_ast(source).validation


def is_string_literal_stmt(stmt: ast.stmt) -> bool:
    """True if `stmt` is a bare string-literal expression (docstring shape)."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def equivalent(a: str, b: str) -> bool:
    """True when docstrings stripped + `ast.unparse` match after parsing.

    Total: unparseable input compares as not-equivalent instead of raising.
    """
    try:
        # Constraint: module-level import would cycle because code transforms
        # compose the analysis enumeration helpers.
        from dr_code.code_transforms import strip_docstrings

        return strip_docstrings(a) == strip_docstrings(b)
    except SyntaxError:
        return False


def module_level_names(tree: ast.Module) -> set[str]:
    """Top-level names that must not be renamed."""
    names: set[str] = set()
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
    """All parameters of `node` in declaration order, including vararg/kwarg."""
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
    """First function definition in `tree` (or `tree` itself); None if absent."""
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
    """Render the signature line, e.g. `def f(x: int) -> str:`."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({args}){returns}:"


def extract_function_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[FunctionArgument]:
    """Named parameters with unparsed annotations; excludes vararg/kwarg."""
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
    """A `FunctionSignatureSite` for every function defined in `tree`."""
    return [
        _function_signature_site(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def annotation_sites(tree: ast.AST) -> list[AnnotationSite]:
    """Every annotation in `tree` as a site (parameter, return, variable)."""
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


def function_locals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Names `node` binds, first-seen order: params, then assignment targets."""
    local_names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        local_names.append(name)

    for arg in function_params(node):
        add(arg.arg)

    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign):
            for name in _target_names(sub.targets):
                add(name)
        elif isinstance(sub, ast.AnnAssign | ast.AugAssign):
            if isinstance(sub.target, ast.Name):
                add(sub.target.id)

    return local_names


def extract_hash_comments(code_str: str) -> list[SourceTextSite]:
    """Every `#` comment in `code_str` as a `SourceTextSite` with location."""
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
    """Every docstring in `tree` as a `SourceTextSite` with owner and location."""
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
    """Hash comments and docstrings joined in source-line order."""
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
