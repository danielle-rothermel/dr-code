"""Structured, total parsing of Markdown-style fenced text.

This module describes document structure only.  It deliberately does not
decide whether fenced or unfenced content is Python, JSON, prose, or a useful
candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


# This pattern is part of recorded metric identities.  Keep its spelling and
# behavior stable even though fence parsing now returns richer values.
FENCE_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?P<fence>```|~~~)(?P<tag>[A-Za-z0-9_+\-]*)[ \t]*$"
)


@dataclass(frozen=True, slots=True)
class FenceDelimiter:
    """One syntactically valid fence-marker line."""

    marker: str
    tag: str


@dataclass(frozen=True, slots=True)
class FenceBlock:
    """A fenced block, including its source-order and closure metadata."""

    index: int
    marker: str
    tag: str
    content: str
    closed: bool


@dataclass(frozen=True, slots=True)
class TextSegment:
    """One source-ordered fenced or unfenced document segment."""

    index: int
    content: str
    fence: FenceBlock | None = None

    @property
    def is_fenced(self) -> bool:
        return self.fence is not None


@dataclass(frozen=True, slots=True)
class FencedDocument:
    """Every nonempty structural segment in source order.

    Empty fenced bodies are retained because their marker, tag, and closure
    state remain meaningful.  Empty unfenced gaps around markers are omitted.
    """

    segments: tuple[TextSegment, ...]

    @property
    def fenced_blocks(self) -> tuple[FenceBlock, ...]:
        return tuple(
            segment.fence
            for segment in self.segments
            if segment.fence is not None
        )

    @property
    def unfenced_segments(self) -> tuple[TextSegment, ...]:
        return tuple(
            segment for segment in self.segments if segment.fence is None
        )


def fence_delimiter(line: str) -> FenceDelimiter | None:
    """Return normalized delimiter metadata for one line, if applicable."""
    match = FENCE_LINE_RE.match(line)
    if match is None:
        return None
    return FenceDelimiter(
        marker=match.group("fence"),
        tag=match.group("tag").casefold(),
    )


def extract_fenced_document(text: str) -> FencedDocument:
    """Parse arbitrary text into ordered fenced and unfenced segments.

    A fence closes only when its marker matches the opener.  A mismatched
    marker is ordinary fenced content, matching the established behavior.
    Unterminated fences retain all remaining text and report ``closed=False``.
    """
    segments: list[TextSegment] = []
    current_lines: list[str] = []
    active: FenceDelimiter | None = None
    fence_index = 0

    def append_unfenced() -> None:
        if not current_lines:
            return
        content = "\n".join(current_lines)
        if content:
            segments.append(TextSegment(index=len(segments), content=content))

    def append_fenced(*, closed: bool) -> None:
        nonlocal fence_index
        assert active is not None
        block = FenceBlock(
            index=fence_index,
            marker=active.marker,
            tag=active.tag,
            content="\n".join(current_lines),
            closed=closed,
        )
        segments.append(
            TextSegment(
                index=len(segments),
                content=block.content,
                fence=block,
            )
        )
        fence_index += 1

    for line in text.split("\n"):
        delimiter = fence_delimiter(line)
        if active is None:
            if delimiter is None:
                current_lines.append(line)
                continue
            append_unfenced()
            current_lines = []
            active = delimiter
            continue

        if delimiter is not None and delimiter.marker == active.marker:
            append_fenced(closed=True)
            current_lines = []
            active = None
            continue
        current_lines.append(line)

    if active is None:
        append_unfenced()
    else:
        append_fenced(closed=False)

    return FencedDocument(segments=tuple(segments))


__all__ = (
    "FENCE_LINE_RE",
    "FenceBlock",
    "FenceDelimiter",
    "FencedDocument",
    "TextSegment",
    "extract_fenced_document",
    "fence_delimiter",
)
