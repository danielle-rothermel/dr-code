from __future__ import annotations

from enum import StrEnum
from typing import Final


SAMPLE_ID_SEP: Final[str] = "::"


class CorruptionName(StrEnum):
    ADD_CODE_FENCES = "add_code_fences"
    ADD_PROSE_WRAPPER = "add_prose_wrapper"
    ADD_SMART_QUOTES = "add_smart_quotes"
    ADD_INDENTATION = "add_indentation"
    ADD_TABS = "add_tabs"
    ADD_TRAILING_WHITESPACE = "add_trailing_whitespace"
    ADD_CRLF = "add_crlf"
    ADD_UNICODE_NOISE = "add_unicode_noise"
    ADD_BLANK_LINES = "add_blank_lines"
    ADD_MARKDOWN_WRAPPERS = "add_markdown_wrappers"
    ADD_INLINE_BACKTICKS = "add_inline_backticks"
    TRUNCATE = "truncate"
    REMOVE_IMPORTS = "remove_imports"
    MANGLE_IMPORT_LINES = "mangle_import_lines"
    DUPLICATE_IMPORTS = "duplicate_imports"
    ADD_MULTIPLE_SOLUTIONS = "add_multiple_solutions"
    ADD_COMMENTS_NOISE = "add_comments_noise"
    ADD_DEAD_CODE = "add_dead_code"
    CHANGE_QUOTE_STYLE = "change_quote_style"
    CHANGE_STRING_FORM = "change_string_form"
    ADD_TYPE_ANNOTATIONS = "add_type_annotations"
    RENAME_LOCALS = "rename_locals"


class TruncationMode(StrEnum):
    MID_FUNCTION = "mid_function"
    MID_LINE = "mid_line"
    MID_STRING = "mid_string"


class MarkdownWrapperMode(StrEnum):
    BLOCKQUOTE = "blockquote"
    NUMBERED_LIST = "numbered_list"
    BULLET_LIST = "bullet_list"


class ImportMangleMode(StrEnum):
    TRAILING_PROSE = "trailing_prose"
    UNBALANCED_PAREN = "unbalanced_paren"
    TRAILING_COMMA = "trailing_comma"


class FenceLangTag(StrEnum):
    PYTHON = "python"
    PY = "py"
    PYTHON3 = "python3"
    NONE = ""
