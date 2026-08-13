from __future__ import annotations

import ast
import copy

from dr_code.core.models import FrozenModel
from dr_code.core.source.python_analysis import (
    FunctionArgument,
    FunctionSignatureSite,
    collect_comments,
    extract_function_signatures,
)
from dr_code.core.source.python_transforms import (
    strip_docstrings_in_tree,
)


class Variable(FrozenModel):
    name: str
    var_type: str | None = None


class FunctionSignature(FrozenModel):
    code_str: str
    signature_str: str
    function_name: str
    function_args: tuple[Variable, ...] = ()


class ParsedCode(FrozenModel):
    code_str: str
    signatures: tuple[FunctionSignature, ...] = ()
    code_without_comments: str = ""
    comments: str = ""
    display_title: str = "ParsedCode"


def _variable_from_argument(argument: FunctionArgument) -> Variable:
    return Variable(name=argument.name, var_type=argument.annotation_source)


def function_signature_from_site(
    site: FunctionSignatureSite,
) -> FunctionSignature:
    return FunctionSignature(
        code_str=ast.unparse(site.owner),
        signature_str=site.signature_source,
        function_name=site.name,
        function_args=tuple(
            _variable_from_argument(argument) for argument in site.arguments
        ),
    )


def remove_comments(tree: ast.AST, *, remove_docstrings: bool = True) -> str:
    if remove_docstrings:
        tree = strip_docstrings_in_tree(copy.deepcopy(tree))
    return ast.unparse(tree)


def parse_code(
    code_str: str, *, display_title: str = "ParsedCode"
) -> ParsedCode:
    tree = ast.parse(code_str)
    signatures = tuple(
        function_signature_from_site(site)
        for site in extract_function_signatures(tree)
    )
    return ParsedCode(
        code_str=code_str,
        signatures=signatures,
        code_without_comments=remove_comments(tree, remove_docstrings=True),
        comments=collect_comments(code_str, tree),
        display_title=display_title,
    )
