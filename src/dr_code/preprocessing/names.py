"""Names for preprocessing pipeline steps."""

from __future__ import annotations

from enum import StrEnum


class StepName(StrEnum):
    """All preprocessing pipeline steps.

    Atomic text steps compose the ``normalize_text`` behavior
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
    # candidate generation
    EXTRACT_CANDIDATES = "extract_candidates"  # settings: strategy tuple
    FIELD_MARKER_EXTRACT = "field_marker_extract"
    # elementwise candidate transforms
    STRIP_FENCES = "strip_fences"
    DEDENT_CANDIDATES = "dedent_candidates"
    SPLIT_ON_NAME_GUARD = "split_on_name_guard"  # drop_if_name
    DROP_AFTER_LAST_RETURN = "drop_after_last_return"
    # import handling — infer_necessary_imports unbundled
    REPAIR_IMPORT_LINES = "repair_import_lines"
    INFER_MISSING_IMPORTS = "infer_missing_imports"
    DEDUPE_IMPORTS = "dedupe_imports"
    # filters and cardinality knobs
    FILTER_COMPILABLE = "filter_compilable"
    FILTER_PLAIN_LITERAL = "filter_plain_literal"
    FILTER_CODE_REPR = "filter_code_repr"
    SELECT_FIRST = "select_first"
    RETURN_ALL = "return_all"


__all__ = ["StepName"]
