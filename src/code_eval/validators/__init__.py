"""Validator registry."""

from typing import Final

from code_eval.names import ValidatorName
from code_eval.validators.ast_parse import AstParseValidator
from code_eval.validators.ast_shape import AstShapeValidator, classify_ast_shape
from code_eval.validators.base import Validator
from code_eval.validators.compile_check import CompileCheckValidator
from code_eval.validators.import_resolve import ImportResolveValidator

VALIDATORS: Final[dict[ValidatorName, type[Validator]]] = {
    ValidatorName.AST_PARSE: AstParseValidator,
    ValidatorName.COMPILE_CHECK: CompileCheckValidator,
    ValidatorName.AST_SHAPE: AstShapeValidator,
    ValidatorName.IMPORT_RESOLVE: ImportResolveValidator,
}

__all__ = [
    "VALIDATORS",
    "AstParseValidator",
    "AstShapeValidator",
    "CompileCheckValidator",
    "ImportResolveValidator",
    "Validator",
    "classify_ast_shape",
]
