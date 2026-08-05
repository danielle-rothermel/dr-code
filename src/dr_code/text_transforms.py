"""Total, best-effort transforms over text that probably contains code.

Functions here accept arbitrary text — raw LLM output, markdown, prose with
embedded code — and never raise; unrepairable input passes through
unchanged. For transforms that assume their input already *is* parseable
Python (and raise `SyntaxError` when it is not), see
`dr_code.code_transforms`.
"""

from __future__ import annotations

import io
import json
import re
import token
import tokenize
import unicodedata
from typing import Final

from dr_code.text_analysis import LINE_SEP, fence_marker

DEFAULT_TAB_WIDTH: Final[int] = 4
FENCE: Final[str] = "```"

MARKDOWN_WRAPPER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[ \t]*(?:>+[ \t]?|\d+[.)][ \t]?|[*+\-][ \t])"
)
BLANK_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
PYTHON_ANCHOR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:def |async def |class |import |from |@|if __name__)"
)

#: ASCII quote -> (left, right) Unicode "smart" counterparts.
SMART_QUOTES: Final[dict[str, tuple[str, str]]] = {
    "'": ("‘", "’"),
    '"': ("“", "”"),
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


def recover_escaped_python(source: str) -> str | None:
    """Recover structurally escaped Python without changing its strings.

    Whole-response JSON strings use JSON's own escaping rules. Other input is
    recovered only after a top-level Python anchor is found at a real or
    escaped line boundary. From that anchor onward, structural ``\\n``,
    ``\\r``, CRLF, and indentation ``\\t`` escapes are decoded only while the
    scanner is outside Python string literals. Ambiguous escapes inside a
    string are preserved, so unsupported shapes fail extraction instead of
    silently changing candidate semantics.
    """
    stripped = source.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            decoded = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            pass
        else:
            if isinstance(decoded, str) and decoded != source:
                return decoded

    anchor_start, opening_fence = _escaped_python_anchor(source)
    if anchor_start is None:
        return None

    recovered, changed = _decode_python_structure(source[anchor_start:])
    if not changed:
        return None
    if opening_fence is not None:
        return f"{opening_fence}\n{recovered}"
    return recovered


def _escaped_python_anchor(source: str) -> tuple[int | None, str | None]:
    line_start = 0
    previous_line: str | None = None
    index = 0
    while True:
        if PYTHON_ANCHOR_RE.match(source, line_start):
            opening_fence = None
            if previous_line is not None:
                marker = fence_marker(previous_line)
                if marker is not None:
                    opening_fence = previous_line
            return line_start, opening_fence

        if index >= len(source):
            return None, None
        break_length = _line_break_length(source, index)
        if break_length is None:
            index += 1
            continue
        previous_line = source[line_start:index]
        line_start = index + break_length
        index = line_start


def _decode_python_structure(source: str) -> tuple[str, bool]:
    decoded_parts: list[str] = []
    quote: str | None = None
    triple_quoted = False
    in_comment = False
    changed = False
    index = 0
    while index < len(source):
        character = source[index]

        if quote is not None:
            if source.startswith("\\", index):
                decoded_parts.append(character)
                index += 1
                if index < len(source):
                    decoded_parts.append(source[index])
                    index += 1
                continue
            closing = quote * (3 if triple_quoted else 1)
            if source.startswith(closing, index):
                decoded_parts.append(closing)
                index += len(closing)
                quote = None
                triple_quoted = False
                continue
            decoded_parts.append(character)
            index += 1
            continue

        break_length = _line_break_length(source, index)
        if break_length is not None:
            decoded_parts.append("\n")
            index += break_length
            in_comment = False
            changed = changed or break_length > 1
            continue

        if _is_unpaired_escape(source, index, "t"):
            decoded_parts.append("\t")
            index += 2
            changed = True
            continue

        if not in_comment and character == "#":
            in_comment = True
        elif not in_comment and character in {'"', "'"}:
            quote = character
            triple_quoted = source.startswith(character * 3, index)
            opening = character * (3 if triple_quoted else 1)
            decoded_parts.append(opening)
            index += len(opening)
            continue

        decoded_parts.append(character)
        index += 1

    return "".join(decoded_parts), changed


def _line_break_length(source: str, index: int) -> int | None:
    if source.startswith("\n", index):
        return 1
    if _is_unpaired_escape(source, index, "r"):
        if _is_unpaired_escape(source, index + 2, "n"):
            return 4
        return 2
    if _is_unpaired_escape(source, index, "n"):
        return 2
    return None


def _is_unpaired_escape(source: str, index: int, escaped: str) -> bool:
    if index < 0 or not source.startswith(f"\\{escaped}", index):
        return False
    preceding_slashes = 0
    cursor = index - 1
    while cursor >= 0 and source[cursor] == "\\":
        preceding_slashes += 1
        cursor -= 1
    return preceding_slashes % 2 == 0


def drop_if_name(text: str) -> list[str]:
    """Split `text` on `if __name__` guard lines, dropping the guards.

    Constraint: a "guard line" is any line whose text contains the substring
    `if __name__` — including comment and string lines — and the split is on
    every occurrence of that line's exact text. A deliberate lossy heuristic
    for LLM-output cleanup, not a syntactic guard detector.
    """
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


def drop_after_last_return(text: str) -> str | None:
    """Truncate `text` after its last complete `return` statement.

    Returns `None` when there is no such boundary to truncate at — no
    `return` inside a function body, or text whose tokenization never
    reaches one. A salvage that cannot be located is not performed, so
    `None` means "nothing to salvage", never "salvaged to nothing".

    The boundary is found by tokenizing rather than by matching lines,
    which is what makes it a *complete* statement: a `NEWLINE` token fires
    only once bracket continuations have closed, so a return spanning
    several lines is kept whole instead of being cut mid-bracket. Tokens
    also distinguish the keyword from the same word inside a string or a
    comment, and `INDENT`/`DEDENT` tracking keeps a module-level `return`
    — which is not a function body's exit — from being treated as one.

    Truncation is lossy and this is a best-effort repair over arbitrary
    text, so it fails closed: a malformed token, an unterminated bracket,
    or inconsistent indentation stops the walk, and only a boundary whose
    own statement had already completed is trusted. A `return` still
    pending when the walk stops yields `None`.
    """
    pending_return = False
    at_statement_start = True
    indent_level = 0
    boundary: tuple[int, int] | None = None
    try:
        for item in tokenize.generate_tokens(io.StringIO(text).readline):
            if item.type == token.ERRORTOKEN and not item.string.isspace():
                break
            if item.type == token.INDENT:
                indent_level += 1
                continue
            if item.type == token.DEDENT:
                indent_level = max(0, indent_level - 1)
                continue
            if item.type in {token.COMMENT, token.NL, token.ENDMARKER}:
                continue
            if item.type == token.NEWLINE or (
                item.type == token.OP and item.string == ";"
            ):
                if pending_return:
                    boundary = item.end
                pending_return = False
                at_statement_start = True
                continue
            if (
                at_statement_start
                and indent_level > 0
                and item.type == token.NAME
                and item.string == "return"
            ):
                pending_return = True
            at_statement_start = False
    except (SyntaxError, ValueError, tokenize.TokenError):
        # ``SyntaxError`` covers ``IndentationError``; ``ValueError`` covers
        # text the tokenizer cannot even encode, such as a lone surrogate.
        pass
    if boundary is None or pending_return:
        return None
    return text[: _source_offset(text, boundary)]


def _source_offset(text: str, position: tuple[int, int]) -> int:
    """The character offset of a 1-based (row, column) token position."""
    row, column = position
    lines = text.splitlines(keepends=True)
    if row > len(lines):
        return len(text)
    return sum(len(line) for line in lines[: row - 1]) + column


__all__ = [
    "DEFAULT_TAB_WIDTH",
    "SMART_QUOTES",
    "collapse_blank_runs",
    "drop_after_last_return",
    "drop_if_name",
    "normalize_line_endings",
    "normalize_text",
    "strip_code_fences",
    "strip_markdown_wrappers",
    "strip_trailing_whitespace",
    "recover_escaped_python",
    "wrap_code_fence",
]
