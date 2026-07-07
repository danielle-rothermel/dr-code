"""Per-repair unit tests.

Each repair is a pure function. We assert:
- the happy path actually fixes the input
- ``RepairResult.changed`` matches whether the source mutated
- ``applied_tags`` contains the right tag(s) when changed=True
- no-ops return changed=False with empty tags
"""

from __future__ import annotations

import ast

from code_eval.names import ImportRepairKind, RepairName
from code_eval.repairs import (
    DedentRepair,
    ImportRepair,
    SmartQuotesRepair,
    TruncationRepair,
)


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


# ---------------------------------------------------------------------------
# SmartQuotes
# ---------------------------------------------------------------------------


def test_smart_quotes_replaces_curly() -> None:
    src = "x = \u201chello\u201d\n"
    result = SmartQuotesRepair().apply(src)
    assert result.changed
    assert result.source == 'x = "hello"\n'
    assert result.applied_tags == (RepairName.SMART_QUOTES.value,)
    assert _parses(result.source)


def test_smart_quotes_noop() -> None:
    result = SmartQuotesRepair().apply('x = "hello"\n')
    assert not result.changed
    assert result.applied_tags == ()


# ---------------------------------------------------------------------------
# Dedent
# ---------------------------------------------------------------------------


def test_dedent_strips_common_indent() -> None:
    src = "    def foo():\n        return 1\n"
    result = DedentRepair().apply(src)
    assert result.changed
    assert result.source.startswith("def foo():")
    assert result.applied_tags == (RepairName.DEDENT.value,)
    assert _parses(result.source)


def test_dedent_noop() -> None:
    result = DedentRepair().apply("def foo():\n    return 1\n")
    assert not result.changed


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_repair_closes_open_block() -> None:
    # Trailing colon with no body — should insert pass.
    src = "def foo():\n    if True:\n"
    result = TruncationRepair().apply(src)
    assert result.changed
    assert _parses(result.source)
    assert result.applied_tags == (RepairName.TRUNCATION.value,)


def test_truncation_repair_trims_dangling_decorator() -> None:
    # Trailing partial decorator is not an EOF/truncation error — repair is a no-op.
    src = "def foo():\n    return 1\n@"
    result = TruncationRepair().apply(src)
    assert not result.changed
    assert result.source == src


def test_truncation_repair_noop_for_valid_input() -> None:
    src = "def foo():\n    return 1\n"
    result = TruncationRepair().apply(src)
    assert not result.changed
    assert result.applied_tags == ()


def test_truncation_repair_noop_for_mid_file_syntax_error() -> None:
    """Mid-file syntax errors must not be 'repaired' by line dropping."""
    src = "def foo():\n    return 1 +\n"
    result = TruncationRepair().apply(src)
    assert not result.changed
    assert result.source == src
    assert result.applied_tags == ()


def test_truncation_repair_recovers_eof_truncation() -> None:
    """True EOF truncation should still trim the dangling tail."""
    src = 'def foo():\n    return "hello'
    result = TruncationRepair().apply(src)
    assert result.changed
    assert _parses(result.source)
    assert result.applied_tags == (RepairName.TRUNCATION.value,)


# ---------------------------------------------------------------------------
# Import recovery
# ---------------------------------------------------------------------------


def test_import_recovery_infers_missing_np() -> None:
    # References np without importing it.
    src = "def foo():\n    return np.array([1])\n"
    result = ImportRepair().apply(src)
    assert result.changed
    assert "import numpy as np" in result.source
    assert RepairName.IMPORT_RECOVERY.value in result.applied_tags
    assert ImportRepairKind.INFERRED.value in result.applied_tags
    assert _parses(result.source)


def test_import_recovery_dedups_duplicate_imports() -> None:
    src = "import math\nimport math\ndef foo(): return math.pi\n"
    result = ImportRepair().apply(src)
    assert result.changed
    # Only one import math should remain.
    assert result.source.count("import math\n") == 1
    assert ImportRepairKind.DEDUP.value in result.applied_tags


def test_import_recovery_infers_missing_np_despite_nested_class_binding() -> None:
    src = "class Config:\n    pd = None\n\nresult = pd.read_csv('f.csv')\n"
    result = ImportRepair().apply(src)
    assert result.changed
    assert "import pandas as pd" in result.source
    assert _parses(result.source)


def test_import_recovery_fixes_mangled_import() -> None:
    # Mangled paren in from-import line.
    src = "from math import (sin, cos\ndef foo(): return sin(0)\n"
    result = ImportRepair().apply(src)
    assert result.applied_tags
    assert _parses(result.source)


def test_import_recovery_noop_for_clean_source() -> None:
    src = "import math\ndef foo(): return math.pi\n"
    result = ImportRepair().apply(src)
    assert not result.changed
    assert result.applied_tags == ()


def test_import_recovery_emits_canonical_tag_first() -> None:
    """When sub-tags fire, the canonical RepairName.IMPORT_RECOVERY tag
    is included alongside (per Phase 2 attribution contract)."""
    src = "def foo(): return np.array([])\n"
    result = ImportRepair().apply(src)
    assert result.applied_tags[0] == RepairName.IMPORT_RECOVERY.value
