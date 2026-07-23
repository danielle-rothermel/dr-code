"""Optional composed identifier-rename tests."""

from __future__ import annotations

import ast

import pytest

from dr_code.mutants.rename import RenameError, rename_entry_point


def test_rename_covers_definition_and_recursive_calls() -> None:
    source = (
        "def fib(n):\n"
        "    if n < 2:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )
    out = rename_entry_point(source, entry_point="fib")
    assert "def target_fxn(n):" in out
    assert "fib" not in out
    assert out.count("target_fxn") == 3  # def + two recursive calls


def test_rename_is_behavior_preserving_ast_shape() -> None:
    source = "def g(x):\n    return x * 2\n"
    out = rename_entry_point(source, entry_point="g", target_name="target_fxn")
    tree = ast.parse(out)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    assert func.name == "target_fxn"
    assert ast.unparse(func.body[0]) == "return x * 2"


def test_rename_missing_entry_point_raises() -> None:
    with pytest.raises(RenameError):
        rename_entry_point("def h(x):\n    return x\n", entry_point="absent")


def test_rename_noop_when_names_match() -> None:
    source = "def target_fxn(x):\n    return x\n"
    assert rename_entry_point(
        source, entry_point="target_fxn", target_name="target_fxn"
    ) == source


def test_rename_leaves_unrelated_names_untouched() -> None:
    source = "def f(f_count):\n    return f_count + 1\n"
    out = rename_entry_point(source, entry_point="f")
    assert "f_count" in out  # the parameter is not the function name
    assert "def target_fxn(f_count):" in out
