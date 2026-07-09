"""Contract and behavior tests for `dr_code.code_transforms`."""

from __future__ import annotations

import pytest

from dr_code.code_transforms import (
    alpha_rename_locals,
    canonicalize,
    dedupe_imports,
    equivalent,
    remove_top_level_imports,
    strip_docstrings,
    strip_type_annotations,
)

UNPARSEABLE = "def broken(:\n"

SOURCE_TO_SOURCE_TRANSFORMS = (
    alpha_rename_locals,
    canonicalize,
    dedupe_imports,
    remove_top_level_imports,
    strip_docstrings,
    strip_type_annotations,
)


@pytest.mark.parametrize(
    "transform",
    SOURCE_TO_SOURCE_TRANSFORMS,
    ids=lambda fn: fn.__name__,
)
def test_code_transforms_raise_syntax_error_on_unparseable_input(
    transform,
) -> None:
    with pytest.raises(SyntaxError):
        transform(UNPARSEABLE)


def test_strip_docstrings_removes_module_function_and_class_docstrings() -> None:
    source = (
        '"""Module doc."""\n'
        "class C:\n"
        '    """Class doc."""\n'
        "    def m(self):\n"
        '        """Method doc."""\n'
        "        return 1\n"
    )
    out = strip_docstrings(source)
    assert '"""' not in out
    assert "return 1" in out


def test_strip_docstrings_keeps_docstring_only_bodies_parseable() -> None:
    out = strip_docstrings('def f():\n    """Only a docstring."""\n')
    assert "pass" in out


def test_equivalent_ignores_formatting_and_docstrings() -> None:
    a = 'def f(x):\n    """Doc."""\n    return (x + 1)\n'
    b = "def f(x):\n    return x + 1\n"
    assert equivalent(a, b)


def test_equivalent_is_false_for_different_code_and_unparseable_input() -> None:
    assert not equivalent("def f():\n    return 1\n", "def f():\n    return 2\n")
    assert not equivalent(UNPARSEABLE, "x = 1\n")


def test_strip_type_annotations_drops_args_returns_and_annassign() -> None:
    source = "def f(x: int, *args: str) -> bool:\n    y: int = 2\n    z: int\n    return x\n"
    out = strip_type_annotations(source)
    assert ":" not in out.split("\n")[0].removesuffix(":")
    assert "def f(x, *args):" in out
    assert "y = 2" in out
    assert "z:" not in out


def test_alpha_rename_locals_renames_params_and_locals_by_default() -> None:
    out = alpha_rename_locals("def f(count):\n    total = count + 1\n    return total\n")
    assert "def f(_v0):" in out
    assert "_v1 = _v0 + 1" in out


def test_alpha_rename_locals_can_preserve_params() -> None:
    out = alpha_rename_locals(
        "def f(count):\n    total = count + 1\n    return total\n",
        rename_params=False,
    )
    assert "def f(count):" in out
    assert "_v0 = count + 1" in out


def test_alpha_rename_locals_preserves_module_level_names() -> None:
    out = alpha_rename_locals("import math\n\ndef f(x):\n    return math.sqrt(x)\n")
    assert "math.sqrt" in out
    assert "def f(" in out


def test_remove_top_level_imports_deletes_only_import_lines() -> None:
    source = "import math\nfrom os import path\n\nx = 1\n"
    assert remove_top_level_imports(source) == "\nx = 1\n"


def test_remove_top_level_imports_no_imports_is_identity() -> None:
    source = "x = 1\n"
    assert remove_top_level_imports(source) == source


def test_dedupe_imports_keeps_first_occurrence_and_trailing_newline() -> None:
    source = "import math\nimport math\nx = 1\n"
    assert dedupe_imports(source) == "import math\nx = 1\n"
