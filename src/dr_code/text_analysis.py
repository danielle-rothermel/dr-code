"""Total analysis helpers for text that probably contains code.

Functions here accept arbitrary text and never raise; unclassifiable input
returns empty blocks or `False`. For transforms over the same best-effort
text boundary, see `dr_code.text_transforms`; for parseable Python source,
see `dr_code.code_analysis` and `dr_code.code_transforms`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

# NOTE: The constants below are consumed by metric operators (currently
# `code_leakage` and `text_stats`), whose recorded values are part of a
# question's metric identity. Changing any of these patterns/sets changes
# those operators' output and therefore requires bumping the affected
# operators' `VERSION`.
FENCE_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?P<fence>```|~~~)(?P<tag>[A-Za-z0-9_+\-]*)[ \t]*$"
)
CODE_ANCHOR_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:def |async def |class |import |from |@|if __name__)"
)
CODE_LIKE_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:\s+\S|"
    r"def |async def |class |import |from |@|if |for |while |with |try |"
    r"except|else|elif|return |raise |pass\b|continue\b|break\b|"
    r"#|"
    r"[a-zA-Z_]\w*\s*=)"
)
WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b\w+\b")
OPERATOR_CHARS: Final[frozenset[str]] = frozenset("+-*/%=<>!&|^~:@")
LINE_SEP: Final[str] = "\n"


def fence_marker(line: str) -> str | None:
    """The fence token (``` or ~~~) if `line` is a fence line, else None."""
    match = FENCE_LINE_RE.match(line)
    if match is None:
        return None
    return match.group("fence")


def is_code_anchor_line(line: str) -> bool:
    """True if `line` starts a Python top-level construct (def/class/import/...)."""
    return bool(CODE_ANCHOR_LINE_RE.match(line))


def is_code_like_line(line: str) -> bool:
    """True if `line` plausibly belongs to a Python code block.

    Blank lines count as code-like so they don't break up a block.
    """
    if not line.strip():
        return True
    return bool(CODE_LIKE_LINE_RE.match(line))


def _append_nonempty(blocks: list[str], lines: list[str]) -> None:
    if lines:
        blocks.append(LINE_SEP.join(lines))


def split_by_fences(text: str) -> tuple[list[str], list[str]]:
    """Split text into unfenced and fenced blocks, dropping fence markers."""
    unfenced: list[str] = []
    fenced: list[str] = []
    current: list[str] = []
    in_fence = False
    active_fence: str | None = None

    for line in text.split(LINE_SEP):
        marker = fence_marker(line)
        if marker is not None and (
            active_fence is None or marker == active_fence
        ):
            _append_nonempty(fenced if in_fence else unfenced, current)
            current = []
            in_fence = not in_fence
            active_fence = marker if in_fence else None
            continue
        current.append(line)

    _append_nonempty(fenced if in_fence else unfenced, current)
    return unfenced, fenced


def candidate_blocks(text: str) -> list[str]:
    """Return fenced blocks when present, otherwise the first unfenced block."""
    unfenced, fenced = split_by_fences(text)
    if fenced:
        return fenced
    return unfenced[:1]


def is_code_like_block(text: str) -> bool:
    """True if `text`'s first line looks like Python (empty text counts)."""
    first_line = text.split(LINE_SEP, 1)[0] if text else None
    if first_line is None:
        return True
    return is_code_like_line(first_line)


def anchored_code_blocks(text: str) -> list[str]:
    """Split `text` into code-like blocks anchored at def/class/import lines."""
    lines = text.split(LINE_SEP)
    if is_code_like_block(text):
        return [text]

    blocks: list[str] = []
    prefix: list[str] = []
    for index, line in enumerate(lines):
        if not is_code_anchor_line(line):
            prefix.append(line)
            continue

        prefix_text = LINE_SEP.join(prefix)
        if prefix and is_code_like_block(prefix_text):
            blocks.append(prefix_text)

        remaining = LINE_SEP.join(lines[index:])
        if is_code_like_block(remaining) or not blocks:
            blocks.append(remaining)
            break
    return blocks


def code_like_blocks(blocks: Iterable[str]) -> list[str]:
    """Flatten `anchored_code_blocks` over every block."""
    code_blocks: list[str] = []
    for block in blocks:
        code_blocks.extend(anchored_code_blocks(block))
    return code_blocks


__all__ = [
    "CODE_ANCHOR_LINE_RE",
    "CODE_LIKE_LINE_RE",
    "FENCE_LINE_RE",
    "LINE_SEP",
    "OPERATOR_CHARS",
    "WORD_RE",
    "candidate_blocks",
    "anchored_code_blocks",
    "code_like_blocks",
    "fence_marker",
    "is_code_anchor_line",
    "is_code_like_block",
    "is_code_like_line",
    "split_by_fences",
]
