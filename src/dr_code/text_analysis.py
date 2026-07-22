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

from dr_code.fenced_text import (
    FENCE_LINE_RE,
    extract_fenced_document,
    fence_delimiter,
)

# NOTE: The constants below are consumed by metric operators (currently
# `code_leakage` and `text_stats`), whose recorded values are part of a
# question's metric identity. Changing any of these patterns/sets changes
# those operators' output and therefore requires bumping the affected
# operators' `VERSION`.
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
    delimiter = fence_delimiter(line)
    return None if delimiter is None else delimiter.marker


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


def split_by_fences(text: str) -> tuple[list[str], list[str]]:
    """Compatibility view over the structured fenced-document parser."""
    document = extract_fenced_document(text)
    unfenced = [
        segment.content
        for segment in document.unfenced_segments
        if segment.content
    ]
    fenced = [
        block.content for block in document.fenced_blocks if block.content
    ]
    if not document.segments and not FENCE_LINE_RE.search(text):
        unfenced.append(text)
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
