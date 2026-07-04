"""Truncation repair.

Attempts to recover from unexpected-EOF parse failures. Strategy:

1. Try the input as-is.
2. If parsing fails with an EOF/unterminated error, walk backwards:
   a. drop trailing whitespace lines.
   b. drop the last (partial) line of code; insert a `pass` if the
      preceding block is now empty.
3. Repeat up to a bounded number of times.

This is intentionally conservative — we never *add* logic, only *trim*
truncated tails or stub them with `pass`.
"""

from __future__ import annotations

import ast
from typing import ClassVar, Final

from code_eval.names import RepairName
from code_eval.repairs.base import Repair, RepairResult

_MAX_ITERATIONS: Final[int] = 80

_EOF_HINTS: Final[tuple[str, ...]] = (
    "unexpected EOF",
    "expected an indented block",
    "unterminated string literal",
    "EOL while scanning",
)


def _looks_truncation_like(err: SyntaxError) -> bool:
    msg = (err.msg or "").lower()
    return any(hint.lower() in msg for hint in _EOF_HINTS)


def _ends_with_open_block(lines: list[str]) -> bool:
    for line in reversed(lines):
        if not line.strip():
            continue
        return line.rstrip().endswith(":")
    return False


def attempt_truncation_repair(source: str) -> tuple[str, bool]:
    """Return (repaired, changed)."""
    try:
        ast.parse(source)
        return source, False
    except SyntaxError as e:
        if not _looks_truncation_like(e):
            return source, False

    lines = source.splitlines()
    # Drop trailing blank lines first.
    while lines and not lines[-1].strip():
        lines.pop()

    for _ in range(_MAX_ITERATIONS):
        candidate = "\n".join(lines)
        # If the last non-blank line ends in `:`, add a `pass` to close it.
        if _ends_with_open_block(lines):
            indent_lines = [ln for ln in lines if ln.strip()]
            last_indent = (
                len(indent_lines[-1]) - len(indent_lines[-1].lstrip()) if indent_lines else 0
            )
            attempt = candidate + "\n" + " " * (last_indent + 4) + "pass"
            try:
                ast.parse(attempt)
                return attempt, attempt != source
            except SyntaxError:
                pass

        try:
            ast.parse(candidate)
            return candidate, candidate != source
        except SyntaxError as e:
            if not lines or not _looks_truncation_like(e):
                break
            lines.pop()

    return source, False


class TruncationRepair(Repair):
    NAME: ClassVar[RepairName] = RepairName.TRUNCATION

    def apply(self, source: str) -> RepairResult:
        fixed, changed = attempt_truncation_repair(source)
        return RepairResult(
            source=fixed,
            applied_tags=(self.NAME.value,) if changed else (),
            changed=changed,
        )
