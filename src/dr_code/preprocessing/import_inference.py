from __future__ import annotations

import ast
import re
import symtable
import textwrap
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
    tree = _parse_or_none(source)
    if tree is None:
        return source

    unresolved = _collect_unresolved_names(source)
    imports = [
        import_statement
        for name, import_statement in IMPORT_ALIAS_MAP.items()
        if name in unresolved
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
    changed = False
    source_lines = source.splitlines()
    lines: list[str] = []
    index = 0
    while index < len(source_lines):
        line = source_lines[index]
        if (
            IMPORT_LINE_RE.match(line)
            and _parse_or_none(line.lstrip()) is None
        ):
            import_end = _valid_multiline_import_end(source_lines, index)
            if import_end is not None:
                lines.extend(source_lines[index:import_end])
                index = import_end
                continue
            fixed = _repair_import_line(line)
            if fixed is not None:
                lines.append(fixed)
            changed = True
            index += 1
            continue
        lines.append(line)
        index += 1
    return "\n".join(lines), changed


def dedupe_import_lines(source: str) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            key = line.rstrip()
            if key in seen:
                continue
            seen.add(key)
        lines.append(line)
    return "\n".join(lines)


def infer_necessary_imports(source: str) -> str:
    repaired, _changed = repair_import_lines(source)
    inferred = infer_missing_imports(repaired)
    return dedupe_import_lines(inferred)


def _parse_or_none(text: str) -> ast.Module | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return ast.parse(text)
        except (SyntaxError, ValueError):
            return None


def _repair_import_line(line: str) -> str | None:
    indentation = line[: len(line) - len(line.lstrip())]
    candidate = TRAILING_JUNK_RE.sub("", line)
    candidate = candidate.strip().rstrip(",")
    if _parse_or_none(candidate) is not None:
        return indentation + candidate

    opens = candidate.count("(")
    closes = candidate.count(")")
    if opens <= closes:
        return None

    closed_candidate = candidate + (")" * (opens - closes))
    if _parse_or_none(closed_candidate) is not None:
        return indentation + closed_candidate
    return None


def _valid_multiline_import_end(lines: list[str], start: int) -> int | None:
    for end in range(start + 2, len(lines) + 1):
        candidate = textwrap.dedent("\n".join(lines[start:end]))
        tree = _parse_or_none(candidate)
        if tree is None:
            continue
        if tree.body and isinstance(tree.body[0], ast.Import | ast.ImportFrom):
            return end
        return None
    return None


def _collect_unresolved_names(source: str) -> set[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            module = symtable.symtable(source, "<import-inference>", "exec")
        except (SyntaxError, ValueError):
            return set()

    module_bound = {
        symbol.get_name()
        for symbol in module.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }
    unresolved: set[str] = set()
    tables = [module]
    while tables:
        table = tables.pop()
        unresolved.update(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_referenced()
            and symbol.is_global()
            and symbol.get_name() not in module_bound
        )
        tables.extend(table.get_children())
    return unresolved


__all__ = [
    "IMPORT_ALIAS_MAP",
    "IMPORT_LINE_RE",
    "TRAILING_JUNK_RE",
    "dedupe_import_lines",
    "infer_missing_imports",
    "infer_necessary_imports",
    "repair_import_lines",
]
