"""Per-normalizer unit tests.

The AST-based normalizers (L0, L1, name_normalize, annotation_strip) are
tested directly with small inputs. The subprocess-based normalizers
(L2-L4, L5, import_sort_dedup, string_form_normalize) are marked
``slow`` and live in ``test_normalizers_subprocess.py``.
"""

from __future__ import annotations

import ast

from code_eval.names import NormalizerName
from code_eval.normalizers import (
    NORMALIZERS,
    AnnotationStrip,
    L0CanonicalAst,
    L1StripCommentsDocstrings,
    NameNormalize,
)


def _parses(s: str) -> bool:
    try:
        ast.parse(s)
        return True
    except SyntaxError:
        return False


def test_registry_has_all_ten_normalizers() -> None:
    expected = {n for n in NormalizerName}
    assert set(NORMALIZERS.keys()) == expected


# ---------------------------------------------------------------------------
# L0 — canonical AST
# ---------------------------------------------------------------------------


def test_l0_round_trips_simple_function() -> None:
    src = "def foo(x):\n    return x + 1\n"
    form = L0CanonicalAst().normalize(src)
    assert form.success
    assert form.normalizer == NormalizerName.L0_CANONICAL_AST
    assert _parses(form.source)
    # Round-trip is idempotent up to whitespace.
    again = L0CanonicalAst().normalize(form.source)
    assert again.source == form.source


def test_l0_failure_on_syntax_error_returns_diagnostic() -> None:
    form = L0CanonicalAst().normalize("def foo(")  # unparseable
    assert not form.success
    assert form.diagnostics
    assert form.diagnostics[0].kind == "l0_parse_error"


# ---------------------------------------------------------------------------
# L1 — strip comments + docstrings
# ---------------------------------------------------------------------------


def test_l1_strips_module_docstring() -> None:
    src = '"""module doc"""\n\ndef foo():\n    return 1\n'
    form = L1StripCommentsDocstrings().normalize(src)
    assert form.success
    assert '"""module doc"""' not in form.source
    assert "def foo" in form.source


def test_l1_strips_function_docstring() -> None:
    src = 'def foo():\n    """fn doc"""\n    return 1\n'
    form = L1StripCommentsDocstrings().normalize(src)
    assert form.success
    assert '"""fn doc"""' not in form.source


def test_l1_strips_comments() -> None:
    src = "def foo():  # inline\n    # block\n    return 1\n"
    form = L1StripCommentsDocstrings().normalize(src)
    assert form.success
    assert "# inline" not in form.source
    assert "# block" not in form.source


def test_l1_empty_function_after_strip_keeps_pass() -> None:
    """Stripping a sole docstring should leave a `pass` so the function
    body remains syntactically valid."""
    src = 'def foo():\n    """only this"""\n'
    form = L1StripCommentsDocstrings().normalize(src)
    assert form.success
    assert _parses(form.source)


# ---------------------------------------------------------------------------
# name_normalize
# ---------------------------------------------------------------------------


def test_name_normalize_renames_args_and_locals() -> None:
    src = "def foo(x, y):\n    z = x + y\n    return z\n"
    form = NameNormalize().normalize(src)
    assert form.success
    assert "_v0" in form.source
    assert _parses(form.source)


def test_name_normalize_preserves_module_level_function_name() -> None:
    src = "def my_func(x):\n    return x\n"
    form = NameNormalize().normalize(src)
    assert form.success
    assert "def my_func(" in form.source


# ---------------------------------------------------------------------------
# annotation_strip
# ---------------------------------------------------------------------------


def test_annotation_strip_drops_args_and_returns() -> None:
    src = "def foo(x: int, y: str = 'a') -> bool:\n    return True\n"
    form = AnnotationStrip().normalize(src)
    assert form.success
    assert ": int" not in form.source
    assert ": str" not in form.source
    assert "-> bool" not in form.source
    assert _parses(form.source)


def test_annotation_strip_converts_annassign() -> None:
    src = "x: int = 5\n"
    form = AnnotationStrip().normalize(src)
    assert form.success
    assert ": int" not in form.source
    assert "x = 5" in form.source
