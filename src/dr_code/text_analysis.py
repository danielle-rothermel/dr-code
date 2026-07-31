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


def anchored_code_blocks(
    text: str, *, segment_prose: bool = False
) -> list[str]:
    """Split `text` into code blocks anchored at def/class/import lines.

    When requested for unfenced text, a non-code-like line becomes a separator
    only when a later anchor follows it. Fenced blocks keep their original
    whole-block and return-salvage behavior.
    """
    lines = text.split(LINE_SEP)
    anchor_indexes = [
        index for index, line in enumerate(lines) if is_code_anchor_line(line)
    ]
    if not anchor_indexes:
        return [text] if is_code_like_block(text) else []

    blocks: list[str] = []
    start = 0 if is_code_like_block(text) else anchor_indexes[0]
    if not segment_prose:
        return [LINE_SEP.join(lines[start:])]

    previous_anchor = anchor_indexes[0]
    for anchor in anchor_indexes[1:]:
        separator = next(
            (
                index
                for index in range(previous_anchor + 1, anchor)
                if not is_code_like_line(lines[index])
            ),
            None,
        )
        if separator is not None:
            blocks.append(LINE_SEP.join(lines[start:separator]))
            start = anchor
        previous_anchor = anchor

    blocks.append(LINE_SEP.join(lines[start:]))
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
