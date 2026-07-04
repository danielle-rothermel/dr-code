"""AST-shape validator.

Passes if the module contains at least one of:

  - FunctionDef / AsyncFunctionDef
  - ClassDef
  - Any executable statement (assignment, expression, for/if/while/with/...)

Fails if the module is empty or contains only a docstring.
"""

from __future__ import annotations

import ast
from typing import ClassVar

from code_eval.models.validation_outcome import ValidationOutcome
from code_eval.names import AstShapeKind, ValidatorName
from code_eval.validators.base import Validator


def classify_ast_shape(source: str) -> AstShapeKind:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return AstShapeKind.EMPTY

    body = tree.body
    if not body:
        return AstShapeKind.EMPTY

    has_func = any(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) for n in body)
    has_class = any(isinstance(n, ast.ClassDef) for n in body)
    has_other = any(
        not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Expr)
        for n in body
    )

    if (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return AstShapeKind.DOCSTRING_ONLY

    if has_func:
        return AstShapeKind.FUNCTION_DEF
    if has_class:
        return AstShapeKind.CLASS_DEF
    if has_other or any(
        isinstance(n, ast.Expr)
        and not (isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
        for n in body
    ):
        return AstShapeKind.EXECUTABLE_STMT

    return AstShapeKind.EMPTY


class AstShapeValidator(Validator):
    NAME: ClassVar[ValidatorName] = ValidatorName.AST_SHAPE

    def validate(self, source: str) -> ValidationOutcome:
        shape = classify_ast_shape(source)
        passed = shape not in {AstShapeKind.EMPTY, AstShapeKind.DOCSTRING_ONLY}
        return ValidationOutcome(
            validator=self.NAME,
            passed=passed,
            ast_shape=shape,
            detail="" if passed else f"shape={shape.value}",
        )
