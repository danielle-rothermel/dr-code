"""Centralized StrEnums, type aliases, and shared constants.

Anything that risks becoming a magic string anywhere in the codebase should
live here. Imports from this module read like `from code_eval.names import
ExtractorName, RepairName, ...` which makes it easy to grep for usage.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# ---------------------------------------------------------------------------
# Version-related constants
# ---------------------------------------------------------------------------

#: Bumped whenever the synthetic corruption recipes change behavior. The
#: dataset JSONL records its build version so regeneration can detect drift.
DATASET_VERSION: Final[str] = "1"

#: Default cache directory, relative to the repo root.
DEFAULT_CACHE_DIR_NAME: Final[str] = ".cache"

#: Default subprocess timeout for external tool invocations (seconds).
DEFAULT_SUBPROCESS_TIMEOUT_S: Final[float] = 5.0

#: Default tab width when normalizing whitespace.
DEFAULT_TAB_WIDTH: Final[int] = 4


# ---------------------------------------------------------------------------
# Tool identifiers
# ---------------------------------------------------------------------------


class ToolName(StrEnum):
    """External tools whose versions are pinned and recorded."""

    PYTHON = "python"
    RUFF = "ruff"
    TY = "ty"
    CODE_EVAL = "code_eval"


# ---------------------------------------------------------------------------
# Pipeline stage names
# ---------------------------------------------------------------------------


class ExtractorName(StrEnum):
    """Identifiers for the extractors that produce candidates from raw input."""

    TEXT_NORMALIZE = "text_normalize"
    DIRECT_PARSE = "direct_parse"
    DIRECT_PARSE_DEDENTED = "direct_parse_dedented"
    FENCES = "fences"
    KEYWORD_ANCHOR = "keyword_anchor"
    PROSE_PATTERNS = "prose_patterns"
    INDENTATION_BLOCK = "indentation_block"
    MARKDOWN_STRIP = "markdown_strip"
    INLINE_SPANS = "inline_spans"


class RepairName(StrEnum):
    """Identifiers for repair passes applied to failed candidates."""

    SMART_QUOTES = "smart_quotes"
    DEDENT = "dedent"
    TRUNCATION = "truncation"
    IMPORT_RECOVERY = "import_recovery"


class ImportRepairKind(StrEnum):
    """Sub-strategies emitted by the import-recovery repair."""

    SYNTACTIC = "import_recovery:syntactic"
    INFERRED = "import_recovery:inferred"
    DEDUP = "import_recovery:dedup"


class ValidatorName(StrEnum):
    """Identifiers for the candidate validators."""

    AST_PARSE = "ast_parse"
    COMPILE_CHECK = "compile_check"
    AST_SHAPE = "ast_shape"
    IMPORT_RESOLVE = "import_resolve"


class NormalizerName(StrEnum):
    """Identifiers for normalizer levels and orthogonal forms."""

    # Layered levels
    L0_CANONICAL_AST = "L0_canonical_ast"
    L1_STRIP_COMMENTS_DOCSTRINGS = "L1_strip_comments_docstrings"
    L2_RUFF_FORMAT = "L2_ruff_format"
    L3_RUFF_FIX_SAFE = "L3_ruff_fix_safe"
    L4_RUFF_FIX_UNSAFE = "L4_ruff_fix_unsafe"
    L5_TY_FIX = "L5_ty_fix"
    # Orthogonal forms
    IMPORT_SORT_DEDUP = "import_sort_dedup"
    NAME_NORMALIZE = "name_normalize"
    ANNOTATION_STRIP = "annotation_strip"
    STRING_FORM_NORMALIZE = "string_form_normalize"


# ---------------------------------------------------------------------------
# Inverse transforms (synthetic corruptions)
# ---------------------------------------------------------------------------


class InverseTransformName(StrEnum):
    """All synthetic corruption transforms.

    Names are paired 1:1 with the expected recovery step. See
    `docs/TESTING.md` for the synthetic corpus contract.
    """

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


# ---------------------------------------------------------------------------
# Severity and diagnostic kinds
# ---------------------------------------------------------------------------


class DiagnosticSeverity(StrEnum):
    """Severity classification for diagnostics emitted along the pipeline."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DiagnosticSource(StrEnum):
    """Source of a diagnostic — useful for filtering downstream."""

    EXTRACTOR = "extractor"
    REPAIR = "repair"
    VALIDATOR = "validator"
    NORMALIZER = "normalizer"
    SUBPROCESS = "subprocess"
    CACHE = "cache"


# ---------------------------------------------------------------------------
# Truncation modes (parameter for the truncate inverse transform)
# ---------------------------------------------------------------------------


class TruncationMode(StrEnum):
    MID_FUNCTION = "mid_function"
    MID_LINE = "mid_line"
    MID_STRING = "mid_string"


# ---------------------------------------------------------------------------
# Markdown wrapper modes
# ---------------------------------------------------------------------------


class MarkdownWrapperMode(StrEnum):
    BLOCKQUOTE = "blockquote"
    NUMBERED_LIST = "numbered_list"
    BULLET_LIST = "bullet_list"


# ---------------------------------------------------------------------------
# Code fence language tags
# ---------------------------------------------------------------------------


class FenceLangTag(StrEnum):
    PYTHON = "python"
    PY = "py"
    PYTHON3 = "python3"
    NONE = ""


# ---------------------------------------------------------------------------
# Frequently-imported names whose absence we try to repair.
# Frozen alias map: short name -> full import statement.
# This is part of the public, auditable contract of the import-repair pass.
# ---------------------------------------------------------------------------

IMPORT_ALIAS_MAP: Final[dict[str, str]] = {
    "np": "import numpy as np",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "torch": "import torch",
    "nn": "import torch.nn as nn",
    "F": "import torch.nn.functional as F",
    "Path": "from pathlib import Path",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "math": "import math",
    "json": "import json",
    "defaultdict": "from collections import defaultdict",
    "Counter": "from collections import Counter",
    "deque": "from collections import deque",
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import field",
    "Enum": "from enum import Enum",
    "StrEnum": "from enum import StrEnum",
    "IntEnum": "from enum import IntEnum",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "itertools": "import itertools",
    "functools": "import functools",
    "reduce": "from functools import reduce",
    "lru_cache": "from functools import lru_cache",
}


# ---------------------------------------------------------------------------
# AST-shape categories
# ---------------------------------------------------------------------------


class AstShapeKind(StrEnum):
    """What the validated module actually contains."""

    FUNCTION_DEF = "function_def"
    CLASS_DEF = "class_def"
    EXECUTABLE_STMT = "executable_stmt"
    DOCSTRING_ONLY = "docstring_only"
    EMPTY = "empty"


# ---------------------------------------------------------------------------
# Recipe / Sample helpers
# ---------------------------------------------------------------------------

#: Separator used in synthetic sample ids: "{task_id}::{recipe_name}".
SAMPLE_ID_SEP: Final[str] = "::"
