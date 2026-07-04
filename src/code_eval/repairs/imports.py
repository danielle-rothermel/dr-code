"""Import-recovery repair.

Three sub-strategies, applied in order:

1. **Syntactic repair of import lines** — strip trailing prose after `#`
   (or other common end-of-line junk), close unbalanced parens, drop
   trailing commas. Drop lines that remain unparseable.
2. **Missing-import inference** — if the module parses but references a
   well-known short name without an import, prepend the conventional
   import from `IMPORT_ALIAS_MAP`. Frozen alias map = auditable.
3. **Duplicate/conflicting import resolution** — dedupe identical import
   lines (keep first); conflicting aliases keep the first and log.

Each successful intervention is logged via the matching
`ImportRepairKind`. The repair is additive — it never removes a name that
the module actually uses.
"""

from __future__ import annotations

import ast
import re
from typing import ClassVar, Final

from code_eval.names import IMPORT_ALIAS_MAP, ImportRepairKind, RepairName
from code_eval.repairs.base import Repair, RepairResult

_IMPORT_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*(import |from )")

_TRAILING_JUNK_RE: Final[re.Pattern[str]] = re.compile(r"\s*(?:#|//|--|/\*).*$")


def _parse_or_none(text: str) -> ast.AST | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _try_fix_import_line(line: str) -> str | None:
    """Best-effort fix for one mangled import line.

    Returns the fixed line if it parses, otherwise None.
    """
    candidate = line
    # 1. Strip trailing prose after a comment-like marker.
    candidate = _TRAILING_JUNK_RE.sub("", candidate)
    candidate = candidate.rstrip().rstrip(",")
    if _parse_or_none(candidate) is not None:
        return candidate
    # 2. Close unbalanced parens.
    opens = candidate.count("(")
    closes = candidate.count(")")
    if opens > closes:
        candidate2 = candidate + (")" * (opens - closes))
        if _parse_or_none(candidate2) is not None:
            return candidate2
    return None


def _step_1_syntactic(source: str) -> tuple[str, bool]:
    """Drop or repair mangled import lines."""
    lines = source.splitlines()
    changed = False
    out_lines: list[str] = []
    for line in lines:
        if _IMPORT_LINE_RE.match(line) and _parse_or_none(line) is None:
            fixed = _try_fix_import_line(line)
            if fixed is not None:
                out_lines.append(fixed)
            # else: drop the unrecoverable import line
            changed = True
            continue
        out_lines.append(line)
    return "\n".join(out_lines), changed


def _collect_referenced_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            # e.g. np.array -> we treat the leftmost root as the name.
            value = node
            while isinstance(value, ast.Attribute):
                value = value.value
            if isinstance(value, ast.Name):
                out.add(value.id)
    return out


def _collect_bound_names(tree: ast.AST) -> set[str]:
    """Names already bound at module scope (assignments + imports)."""
    if not isinstance(tree, ast.Module):
        return set()
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound.add(node.name)
    return bound


def _step_2_infer(source: str) -> tuple[str, bool]:
    tree = _parse_or_none(source)
    if tree is None:
        return source, False
    referenced = _collect_referenced_names(tree)
    bound = _collect_bound_names(tree)
    to_add: list[str] = []
    for short, full in IMPORT_ALIAS_MAP.items():
        if short in referenced and short not in bound:
            to_add.append(full)
    if not to_add:
        return source, False
    prefix = "\n".join(to_add) + "\n"
    return prefix + source, True


def _step_3_dedup(source: str) -> tuple[str, bool]:
    """Drop exact-duplicate import lines (keep first)."""
    seen: set[str] = set()
    out_lines: list[str] = []
    changed = False
    for line in source.splitlines():
        if _IMPORT_LINE_RE.match(line):
            key = line.strip()
            if key in seen:
                changed = True
                continue
            seen.add(key)
        out_lines.append(line)
    return "\n".join(out_lines), changed


def attempt_import_repair(source: str) -> tuple[str, tuple[str, ...]]:
    """Return (repaired_source, tuple of applied tags)."""
    tags: list[str] = []
    src = source
    src, changed1 = _step_1_syntactic(src)
    if changed1:
        tags.append(ImportRepairKind.SYNTACTIC.value)
    src, changed2 = _step_2_infer(src)
    if changed2:
        tags.append(ImportRepairKind.INFERRED.value)
    src, changed3 = _step_3_dedup(src)
    if changed3:
        tags.append(ImportRepairKind.DEDUP.value)
    return src, tuple(tags)


class ImportRepair(Repair):
    NAME: ClassVar[RepairName] = RepairName.IMPORT_RECOVERY

    def apply(self, source: str) -> RepairResult:
        fixed, sub_tags = attempt_import_repair(source)
        # The high-level repair name plus any sub-strategy tags.
        applied: tuple[str, ...] = ((self.NAME.value, *sub_tags)) if sub_tags else ()
        return RepairResult(
            source=fixed,
            applied_tags=applied,
            changed=bool(sub_tags),
        )
