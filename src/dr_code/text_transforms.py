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

DEFAULT_TAB_WIDTH: Final[int] = 4
FENCE: Final[str] = "```"

FENCE_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?P<fence>```|~~~)(?P<tag>[A-Za-z0-9_+\-]*)[ \t]*$"
)
MARKDOWN_WRAPPER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?:>+[ \t]?|\d+[.)][ \t]?|[*+\-][ \t])"
)
BLANK_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
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


def fence_marker(line: str) -> str | None:
    """The fence token (``` or ~~~) if `line` is a fence line, else None."""
    match = FENCE_LINE_RE.match(line)
    if match is None:
        return None
    return match.group("fence")


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


__all__ = [
    "DEFAULT_TAB_WIDTH",
    "SMART_QUOTES",
    "collapse_blank_runs",
    "fence_marker",
    "is_code_anchor_line",
    "is_code_like_line",
    "normalize_line_endings",
    "normalize_smart_quotes",
    "normalize_text",
    "strip_code_fences",
    "strip_markdown_wrappers",
    "strip_trailing_whitespace",
    "wrap_code_fence",
]
