"""Total, best-effort transforms over text that probably contains code.

Functions here accept arbitrary text — raw LLM output, markdown, prose with
embedded code — and never raise; unrepairable input passes through
unchanged. For transforms that assume their input already *is* parseable
Python (and raise `SyntaxError` when it is not), see
`dr_code.code_transforms`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from dr_code.text_analysis import fence_marker

DEFAULT_TAB_WIDTH: Final[int] = 4
FENCE: Final[str] = "```"
LINE_SEP: Final[str] = "\n"

MARKDOWN_WRAPPER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?:>+[ \t]?|\d+[.)][ \t]?|[*+\-][ \t])"
)
BLANK_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
RETURN_LINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*return(?:\b|$)")

#: ASCII quote -> (left, right) Unicode "smart" counterparts.
SMART_QUOTES: Final[dict[str, tuple[str, str]]] = {
    "'": ("‘", "’"),
    '"': ("“", "”"),
}
_SMART_QUOTE_TRANSLATION: Final[dict[int, str]] = {
    ord(smart): ascii_quote
    for ascii_quote, pair in SMART_QUOTES.items()
    for smart in pair
}


def normalize_line_endings(source: str) -> str:
    """Convert CRLF and bare CR line endings to LF."""
    return source.replace("\r\n", "\n").replace("\r", "\n")


def strip_trailing_whitespace(source: str) -> str:
    """Strip trailing whitespace from every LF-separated line.

    CR characters count as trailing whitespace, so CRLF input comes back
    with LF endings.
    """
    return "\n".join(line.rstrip() for line in source.split("\n"))


def collapse_blank_runs(source: str) -> str:
    """Collapse runs of three or more newlines down to one blank line."""
    return BLANK_RUN_RE.sub("\n\n", source)


def normalize_text(source: str, tab_width: int = DEFAULT_TAB_WIDTH) -> str:
    """Canonical text cleanup: LF endings, NFKC, tabs expanded, trailing
    whitespace stripped, blank runs collapsed, outer newlines trimmed."""
    text = normalize_line_endings(source)
    text = unicodedata.normalize("NFKC", text)
    text = text.expandtabs(tab_width)
    text = strip_trailing_whitespace(text)
    text = collapse_blank_runs(text)
    return text.strip("\n")


def strip_code_fences(source: str) -> str:
    """Drop a leading and/or trailing fence line wrapping the source."""
    lines = source.split("\n")
    trailing_newline = bool(lines) and lines[-1] == ""
    if trailing_newline:
        lines = lines[:-1]
    if lines and fence_marker(lines[0]) is not None:
        lines = lines[1:]
    if lines and fence_marker(lines[-1]) is not None:
        lines = lines[:-1]
    return "\n".join(lines) + ("\n" if trailing_newline else "")


def wrap_code_fence(source: str, lang: str = "python") -> str:
    """Wrap the source in a markdown code fence, tagged unless `lang` is empty."""
    opening = f"{FENCE}{lang}" if lang else FENCE
    return f"{opening}\n{source.rstrip()}\n{FENCE}\n"


def strip_markdown_wrappers(source: str) -> str:
    """Remove one leading blockquote/list/bullet marker from every line."""
    return "\n".join(
        MARKDOWN_WRAPPER_RE.sub("", line, count=1)
        for line in source.splitlines()
    )


def normalize_smart_quotes(source: str) -> str:
    """Replace Unicode "smart" quotes with their ASCII counterparts."""
    return source.translate(_SMART_QUOTE_TRANSLATION)


def drop_if_name(text: str) -> list[str]:
    lines = text.split(LINE_SEP)
    split_lines = [line for line in lines if "if __name__" in line]
    if not split_lines:
        return [text]

    remaining = text
    splits: list[str] = []
    for split_line in split_lines:
        before, *after = remaining.split(split_line)
        splits.append(before)
        if after:
            remaining = LINE_SEP.join(after)
    return splits


def drop_after_last_return(text: str) -> str:
    lines = text.split(LINE_SEP)
    for index in range(len(lines) - 1, -1, -1):
        if RETURN_LINE_RE.match(lines[index]):
            return LINE_SEP.join(lines[: index + 1])
    return text


__all__ = [
    "DEFAULT_TAB_WIDTH",
    "SMART_QUOTES",
    "collapse_blank_runs",
    "drop_after_last_return",
    "drop_if_name",
    "normalize_line_endings",
    "normalize_smart_quotes",
    "normalize_text",
    "strip_code_fences",
    "strip_markdown_wrappers",
    "strip_trailing_whitespace",
    "wrap_code_fence",
]
