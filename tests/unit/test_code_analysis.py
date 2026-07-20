"""Contract and behavior tests for `dr_code.code_analysis`."""

from __future__ import annotations

import ast

import pytest

from dr_code.code_analysis import (
    AnnotationKind,
    TextSiteKind,
    annotation_sites,
    collect_comments,
    equivalent,
    extract_docstrings,
    extract_function_args,
    extract_function_signatures,
    extract_hash_comments,
    find_function_node,
    format_function_signature,
    function_locals,
    function_params,
    is_cpython_parser_stack_overflow,
    module_level_names,
    top_level_import_linenos,
    validate_python,
    validate_python_source,
    validate_python_source_with_ast,
)

UNPARSEABLE = "def broken(:\n"


def test_validate_python_raises_syntax_error_on_unparseable_input() -> None:
    with pytest.raises(SyntaxError):
        validate_python(UNPARSEABLE)


def test_validate_python_source_is_total_and_reports_errors() -> None:
    validation = validate_python_source(UNPARSEABLE)

    assert validation.parse_ok is False
    assert validation.compile_ok is False
    assert validation.parse_error is not None
    assert validation.compile_error is not None


def test_validate_python_source_with_ast_returns_reusable_tree() -> None:
    validated = validate_python_source_with_ast("x = 1\n")
    assert validated.validation.parse_ok is True
    assert isinstance(validated.tree, ast.Module)

    validated = validate_python_source_with_ast(UNPARSEABLE)
    assert validated.validation.parse_ok is False
    assert validated.tree is None


def test_validate_python_source_classifies_only_parser_stack_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser_stack_overflow(source: str) -> ast.Module:
        raise MemoryError("Parser stack overflowed")

    monkeypatch.setattr(ast, "parse", parser_stack_overflow)
    validated = validate_python_source("x = 1\n")

    assert validated.parser_stack_overflow is True
    assert validated.parse_ok is False
    assert validated.compile_ok is False
    assert validated.parse_error == "MemoryError: Parser stack overflowed"
    assert is_cpython_parser_stack_overflow(
        MemoryError("Parser stack overflowed - source too complex")
    )
    assert not is_cpython_parser_stack_overflow(MemoryError("out of memory"))


def test_validate_python_source_reraises_unrelated_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def out_of_memory(source: str) -> ast.Module:
        raise MemoryError("allocation failed")

    monkeypatch.setattr(ast, "parse", out_of_memory)
    with pytest.raises(MemoryError, match="allocation failed"):
        validate_python_source("x = 1\n")


def test_validate_python_source_records_parser_recursion_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def parser_recursion(source: str) -> ast.Module:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(ast, "parse", parser_recursion)
    validated = validate_python_source("x = 1\n")

    assert validated.parser_stack_overflow is False
    assert validated.parser_recursion_overflow is True
    assert validated.parse_ok is False
    assert validated.compile_ok is False
    assert validated.parse_error == (
        "RecursionError: maximum recursion depth exceeded"
    )


def test_validate_python_source_captures_syntax_warnings() -> None:
    source = "def f():\n    return '" + chr(92) + "q'\n"
    validated = validate_python_source(source)

    assert validated.compile_ok is True
    assert validated.compile_warnings
    assert validated.compile_warnings[0].startswith("SyntaxWarning:")


def test_validate_python_source_preserves_warning_source_locations() -> None:
    invalid_escape = chr(92) + "q"
    source = f"a = '{invalid_escape}'\nb = '{invalid_escape}'\n"

    validated = validate_python_source(source)

    assert len(validated.compile_warnings) == 2
    assert validated.compile_warnings[0].endswith("(line 1)")
    assert validated.compile_warnings[1].endswith("(line 2)")


def test_validate_python_source_preserves_repeated_warnings_on_one_line() -> (
    None
):
    validated = validate_python_source("a = x is 1; b = y is 1\n")

    assert len(validated.compile_warnings) == 2
    assert all(
        warning.endswith("(line 1)") for warning in validated.compile_warnings
    )


def test_equivalent_ignores_formatting_and_docstrings() -> None:
    a = 'def f(x):\n    """Doc."""\n    return (x + 1)\n'
    b = "def f(x):\n    return x + 1\n"
    assert equivalent(a, b)


def test_equivalent_is_false_for_different_code_and_unparseable_input() -> (
    None
):
    assert not equivalent(
        "def f():\n    return 1\n", "def f():\n    return 2\n"
    )
    assert not equivalent(UNPARSEABLE, "x = 1\n")


def test_module_level_names_collects_current_public_contract() -> None:
    tree = ast.parse(
        "import math as m\n"
        "from os import path\n"
        "value = 1\n"
        "class C: pass\n"
        "def f(): pass\n"
    )

    assert module_level_names(tree) == {"m", "path", "value", "C", "f"}


def test_top_level_import_linenos_covers_multiline_imports() -> None:
    tree = ast.parse("from os import (\n    path,\n)\nvalue = 1\n")

    assert top_level_import_linenos(tree) == {1, 2, 3}


def test_annotation_sites_include_source_location_and_value_flag() -> None:
    tree = ast.parse(
        "def f(x: int) -> str:\n"
        "    y: list[int] = []\n"
        "    z: float\n"
        "    return str(x)\n"
    )

    sites = annotation_sites(tree)

    assert [
        (site.kind, site.name, site.annotation_source) for site in sites
    ] == [
        (AnnotationKind.PARAMETER, "x", "int"),
        (AnnotationKind.RETURN, None, "str"),
        (AnnotationKind.VARIABLE, "y", "list[int]"),
        (AnnotationKind.VARIABLE, "z", "float"),
    ]
    assert sites[0].location.lineno == 1
    assert sites[0].location.col_offset > 0
    assert sites[2].has_value is True
    assert sites[3].has_value is False


def test_function_params_and_locals_preserve_discovery_order() -> None:
    tree = ast.parse(
        "def f(x, /, y, *args, z, **kwargs):\n"
        "    total = x + y\n"
        "    extra: int = 1\n"
        "    total += extra\n"
    )
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)

    assert [arg.arg for arg in function_params(function)] == [
        "x",
        "y",
        "z",
        "args",
        "kwargs",
    ]
    assert function_locals(function) == [
        "x",
        "y",
        "z",
        "args",
        "kwargs",
        "total",
        "extra",
    ]


def test_function_signature_helpers_return_plain_analysis_values() -> None:
    tree = ast.parse(
        "async def f(x: int, *args: str, y: bool = False, **kwargs) -> str:\n"
        "    return str(x)\n"
    )
    function = find_function_node(tree)
    assert isinstance(function, ast.AsyncFunctionDef)

    signatures = extract_function_signatures(tree)

    assert format_function_signature(function) == (
        "async def f(x: int, *args: str, y: bool=False, **kwargs) -> str:"
    )
    assert [
        (arg.name, arg.annotation_source)
        for arg in extract_function_args(function)
    ] == [
        ("x", "int"),
        ("y", "bool"),
    ]
    assert signatures[0].name == "f"
    assert signatures[0].signature_source.startswith("async def f(")
    assert signatures[0].location.lineno == 1
    assert signatures[0].owner is function


def test_comment_and_docstring_sites_share_text_site_shape() -> None:
    source = (
        "# module comment\n"
        '"""module doc"""\n'
        "def f():\n"
        '    """function doc"""\n'
        "    return 1  # inline comment\n"
    )
    tree = ast.parse(source)

    hash_comments = extract_hash_comments(source)
    docstrings = extract_docstrings(tree)

    assert [
        (site.kind, site.text, site.location.lineno) for site in hash_comments
    ] == [
        (TextSiteKind.HASH_COMMENT, "module comment", 1),
        (TextSiteKind.HASH_COMMENT, "inline comment", 5),
    ]
    assert [(site.kind, site.name, site.text) for site in docstrings] == [
        (TextSiteKind.DOCSTRING, None, "module doc"),
        (TextSiteKind.DOCSTRING, "f", "function doc"),
    ]
    assert collect_comments(source, tree) == (
        "module comment\nmodule doc\nfunction doc\ninline comment"
    )
