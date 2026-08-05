"""Infer, repair, and dedupe import lines for extracted code candidates.

Public step bodies behind the ``repair_import_lines`` /
``infer_missing_imports`` / ``dedupe_imports`` steps, plus
``infer_necessary_imports``, which runs all three in one call. Names in
``IMPORT_ALIAS_MAP`` are injected only when referenced and not bound anywhere
in the candidate's syntax tree — a conservative rule, since injecting a wrong
import is worse than skipping one. Injected imports follow the module header
a candidate already carries: a leading docstring, then contiguous
``from __future__`` imports.
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Final

IMPORT_ALIAS_MAP: Final[dict[str, str]] = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "torch": "import torch",
    "nn": "import torch.nn as nn",
    "F": "import torch.nn.functional as F",
    "Path": "from pathlib import Path",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "math": "import math",
    "json": "import json",
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "deque": "from collections import deque",
    "Enum": "from enum import Enum",
    "StrEnum": "from enum import StrEnum",
    "IntEnum": "from enum import IntEnum",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "itertools": "import itertools",
    "functools": "import functools",
    "reduce": "from functools import reduce",
    "lru_cache": "from functools import lru_cache",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
    "Set": "from typing import Set",
    "Optional": "from typing import Optional",
    "Union": "from typing import Union",
    "Any": "from typing import Any",
}

IMPORT_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(import |from )")
TRAILING_JUNK_RE: Final[re.Pattern[str]] = re.compile(r"\s*(?:#|//|--|/\*).*$")


def infer_missing_imports(source: str) -> str:
    """Insert imports for mapped names referenced but never bound.

    Parses ``source``; for each ``IMPORT_ALIAS_MAP`` name referenced but not
    bound anywhere in the tree, inserts its import statement at the earliest
    line that keeps the module's own header intact — see
    ``_inferred_import_insertion_line``. Unparseable input passes through
    unchanged.
    """
    tree = _parse_or_none(source)
    if tree is None:
        return source

    referenced = _collect_referenced_names(tree)
    bound = _collect_bound_names(tree)
    imports = [
        import_statement
        for name, import_statement in IMPORT_ALIAS_MAP.items()
        if name in referenced and name not in bound
    ]
    if not imports:
        return source
    import_block = "\n".join(imports) + "\n"
    insertion_line = _inferred_import_insertion_line(tree)
    if insertion_line == 0:
        return import_block + source
    lines = source.splitlines(keepends=True)
    offset = sum(len(line) for line in lines[:insertion_line])
    return source[:offset] + import_block + source[offset:]


def _inferred_import_insertion_line(tree: ast.Module) -> int:
    """The line inferred imports go after: header first, then imports.

    Two leading constructs own the top of a module and cannot be displaced.
    A ``from __future__`` import is only legal before any other statement,
    so an import placed above one compiles nowhere and the candidate is
    dropped by the compilability filter. A leading string expression is the
    module docstring only while it is the first statement; an import above
    it demotes it to a bare expression and the module loses its ``__doc__``.

    Returns the 1-based line the import block follows — the end of a
    leading docstring plus any contiguous ``from __future__`` imports after
    it — or ``0`` when the module has neither and the block goes first.
    """
    body = tree.body
    index = 0
    insertion_line = 0
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            insertion_line = body[0].end_lineno or body[0].lineno
            index = 1
    while index < len(body):
        statement = body[index]
        if not (
            isinstance(statement, ast.ImportFrom)
            and statement.module == "__future__"
        ):
            break
        insertion_line = statement.end_lineno or statement.lineno
        index += 1
    return insertion_line


def repair_import_lines(source: str) -> tuple[str, bool]:
    """Repair structurally broken import lines; return (repaired, changed)."""
    changed = False
    lines: list[str] = []
    for line in source.splitlines():
        if IMPORT_LINE_RE.match(line) and _parse_or_none(line) is None:
            fixed = _repair_import_line(line)
            if fixed is not None:
                lines.append(fixed)
            changed = True
            continue
        lines.append(line)
    return "\n".join(lines), changed


def dedupe_import_lines(source: str) -> str:
    """Drop later duplicate import lines, keyed on the stripped line."""
    seen: set[str] = set()
    lines: list[str] = []
    for line in source.splitlines():
        if IMPORT_LINE_RE.match(line):
            key = line.strip()
            if key in seen:
                continue
            seen.add(key)
        lines.append(line)
    return "\n".join(lines)


def infer_necessary_imports(source: str) -> str:
    """Repair, infer, then dedupe imports — the full pass in one call."""
    repaired, _changed = repair_import_lines(source)
    inferred = infer_missing_imports(repaired)
    return dedupe_import_lines(inferred)


def _parse_or_none(text: str) -> ast.Module | None:
    """Parse ``text``, or ``None`` when it is not parseable Python.

    ``ValueError`` is caught alongside ``SyntaxError`` because text that
    cannot be encoded at all — a lone surrogate, a null byte — raises from
    ``ast.parse`` rather than failing to parse. Such text is unusable
    input, not a repair opportunity, so it passes through untouched and is
    rejected by the compilability filter, matching
    ``code_analysis.validate_python_source_with_ast``.

    ``SyntaxWarning`` is suppressed because this parse is speculative:
    structural cleaning asks whether text is parseable, it does not own
    diagnostic facts about it. Inspection later parses the surviving source
    and records its diagnostics in the trace, so letting a warning escape
    here only duplicates that record against text that may not survive.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.parse(text)
        except (SyntaxError, ValueError):
            return None


def _repair_import_line(line: str) -> str | None:
    candidate = TRAILING_JUNK_RE.sub("", line)
    candidate = candidate.rstrip().rstrip(",")
    if _parse_or_none(candidate) is not None:
        return candidate

    opens = candidate.count("(")
    closes = candidate.count(")")
    if opens <= closes:
        return None

    closed_candidate = candidate + (")" * (opens - closes))
    if _parse_or_none(closed_candidate) is not None:
        return closed_candidate
    return None


def _collect_referenced_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            value = node
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                names.add(value.id)
    return names


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """Names bound anywhere in the tree — conservative against injection.

    A name bound by any construct at any depth (parameters, assignments,
    imports, comprehension targets, ``for``/``with``/``except``/``match``
    bindings, ``def``/``class`` names, ``global``/``nonlocal`` declarations)
    must never be injected, even if a mapped alias shares its spelling.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(
            node.ctx, ast.Store | ast.Del
        ):
            names.add(node.id)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.MatchStar) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
        elif isinstance(node, ast.MatchAs) and node.name is not None:
            names.add(node.name)
    return names


__all__ = [
    "IMPORT_ALIAS_MAP",
    "IMPORT_LINE_RE",
    "TRAILING_JUNK_RE",
    "dedupe_import_lines",
    "infer_missing_imports",
    "infer_necessary_imports",
    "repair_import_lines",
]
