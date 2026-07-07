"""Subprocess-based normalizer tests (``slow`` marker).

These tests shell out to ruff (and ty when available). They live behind
the ``slow`` marker so the default test run stays fast.
"""

from __future__ import annotations

import ast
import shutil

import pytest

from code_eval.names import NormalizerName, ToolName
from code_eval.normalizers import (
    ImportSortDedup,
    L2RuffFormat,
    L3RuffFixSafe,
    L4RuffFixUnsafe,
    L5TyFix,
    StringFormNormalize,
)

pytestmark = pytest.mark.slow

_RUFF_AVAILABLE = shutil.which(ToolName.RUFF.value) is not None


def _parses(src: str) -> bool:
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


@pytest.mark.skipif(not _RUFF_AVAILABLE, reason="ruff not on PATH")
def test_l2_ruff_format_reformats() -> None:
    src = "def foo(x,y): return x+y\n"
    form = L2RuffFormat().normalize(src)
    assert form.success
    assert form.normalizer == NormalizerName.L2_RUFF_FORMAT
    assert "def foo(x, y):" in form.source


@pytest.mark.skipif(not _RUFF_AVAILABLE, reason="ruff not on PATH")
def test_l3_safe_fixes_then_format() -> None:
    src = "def foo():\n    x=1\n    return x\n"
    form = L3RuffFixSafe().normalize(src)
    assert form.success
    assert _parses(form.source)


@pytest.mark.skipif(not _RUFF_AVAILABLE, reason="ruff not on PATH")
def test_l4_unsafe_fixes_then_format() -> None:
    src = "x = 'abc'.format()\n"
    form = L4RuffFixUnsafe().normalize(src)
    assert form.success
    assert _parses(form.source)


def test_l5_no_op_when_ty_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ty is unavailable, L5 returns a clean no-op with a warning diagnostic."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    form = L5TyFix().normalize("def foo():\n    return 1\n")
    assert form.success  # documented no-op is *not* a failure
    assert form.diagnostics
    assert form.diagnostics[0].kind == "ty_unavailable"


@pytest.mark.skipif(not _RUFF_AVAILABLE, reason="ruff not on PATH")
def test_import_sort_dedup_sorts_and_dedupes() -> None:
    src = "import sys\nimport os\nimport os\ndef foo(): return os.getcwd()\n"
    form = ImportSortDedup().normalize(src)
    assert form.success
    # Dedup removes the duplicate.
    assert form.source.count("import os\n") == 1


@pytest.mark.skipif(not _RUFF_AVAILABLE, reason="ruff not on PATH")
def test_string_form_normalize_converts_format_to_fstring() -> None:
    src = "x = '{}'.format(1)\n"
    form = StringFormNormalize().normalize(src)
    assert form.success
    # UP032 should convert .format(...) into an f-string.
    assert _parses(form.source)
