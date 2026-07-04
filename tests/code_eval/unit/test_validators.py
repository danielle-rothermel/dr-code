"""Unit tests for validators."""

from __future__ import annotations

from code_eval.names import AstShapeKind, ValidatorName
from code_eval.validators import (
    AstParseValidator,
    AstShapeValidator,
    CompileCheckValidator,
    ImportResolveValidator,
    classify_ast_shape,
)


def test_ast_parse_passes_valid_source() -> None:
    outcome = AstParseValidator().validate("def foo():\n    return 1\n")
    assert outcome.passed
    assert outcome.validator == ValidatorName.AST_PARSE


def test_ast_parse_fails_on_syntax_error() -> None:
    outcome = AstParseValidator().validate("def foo(\n")
    assert not outcome.passed
    assert outcome.detail


def test_compile_check_passes_valid_source() -> None:
    outcome = CompileCheckValidator().validate("x = 1\n")
    assert outcome.passed


def test_compile_check_rejects_null_bytes() -> None:
    outcome = CompileCheckValidator().validate("x = '\x00'\n")
    assert not outcome.passed


def test_ast_shape_accepts_function_def() -> None:
    source = "def foo():\n    return 1\n"
    assert classify_ast_shape(source) == AstShapeKind.FUNCTION_DEF
    outcome = AstShapeValidator().validate(source)
    assert outcome.passed


def test_ast_shape_rejects_docstring_only() -> None:
    source = '"""only a docstring"""\n'
    outcome = AstShapeValidator().validate(source)
    assert not outcome.passed
    assert outcome.ast_shape == AstShapeKind.DOCSTRING_ONLY


def test_import_resolve_passes_for_stdlib() -> None:
    outcome = ImportResolveValidator().validate("import math\n")
    assert outcome.passed


def test_import_resolve_fails_for_missing_module() -> None:
    outcome = ImportResolveValidator().validate("import definitely_not_a_real_module_xyz\n")
    assert not outcome.passed
    assert "missing" in (outcome.detail or "")
