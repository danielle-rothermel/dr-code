"""Names for preprocessing pipeline steps."""

from __future__ import annotations

from enum import StrEnum


class StepName(StrEnum):
    """All preprocessing pipeline steps.

    Atomic text steps are the constituents of today's ``normalize_text``
    (plus smart quotes as its own step). Candidate-generation steps emit
    ``CodeCandidateSetArtifact`` from text. Elementwise steps map over a
    candidate set. Filters keep/drop candidates; cardinality knobs fix the
    output kind.
    """

    # atomic text steps — the constituents of normalize_text, plus smart quotes
    NORMALIZE_LINE_ENDINGS = "normalize_line_endings"
    NORMALIZE_UNICODE = "normalize_unicode"  # NFKC
    NORMALIZE_SMART_QUOTES = "normalize_smart_quotes"
    EXPAND_TABS = "expand_tabs"  # settings: tab_width
    STRIP_TRAILING_WHITESPACE = "strip_trailing_whitespace"
    COLLAPSE_BLANK_RUNS = "collapse_blank_runs"
    TRIM_OUTER_BLANKS = "trim_outer_blanks"
    REQUIRE_NONBLANK_TEXT = "require_nonblank_text"
    # candidate generation
    EXTRACT_CANDIDATES = "extract_candidates"  # settings: strategy tuple
    # elementwise candidate transforms
    STRIP_FENCES = "strip_fences"
    DEDENT_CANDIDATES = "dedent_candidates"  # in every best-effort def
    SPLIT_ON_NAME_GUARD = "split_on_name_guard"  # drop_if_name
    DROP_AFTER_LAST_RETURN = "drop_after_last_return"
    EXPAND_LAST_RETURN_SALVAGE = "expand_last_return_salvage"
    # import handling — infer_necessary_imports unbundled
    REPAIR_IMPORT_LINES = "repair_import_lines"
    INFER_MISSING_IMPORTS = "infer_missing_imports"
    DEDUPE_IMPORTS = "dedupe_imports"
    # filters and cardinality knobs
    FILTER_NONBLANK_CANDIDATES = "filter_nonblank_candidates"
    DEDUPE_CANDIDATES = "dedupe_candidates"
    IDENTIFY_CANDIDATES = "identify_candidates"
    FILTER_COMPILABLE = "filter_compilable"
    FILTER_PLAIN_LITERAL = "filter_plain_literal"
    FILTER_CODE_REPR = "filter_code_repr"
    FILTER_HAS_TOP_LEVEL_FUNCTION = "filter_has_top_level_function"
    MATERIALIZE_CANDIDATES = "materialize_candidates"
    SELECT_FIRST = "select_first"
    RETURN_ALL = "return_all"


__all__ = ["StepName"]
