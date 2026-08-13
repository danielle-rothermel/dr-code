from __future__ import annotations

from enum import StrEnum


class StepName(StrEnum):
    NORMALIZE_LINE_ENDINGS = "normalize_line_endings"
    NORMALIZE_UNICODE = "normalize_unicode"
    EXPAND_TABS = "expand_tabs"
    STRIP_TRAILING_WHITESPACE = "strip_trailing_whitespace"
    COLLAPSE_BLANK_RUNS = "collapse_blank_runs"
    TRIM_OUTER_BLANKS = "trim_outer_blanks"
    REJECT_BLANK_INPUT = "reject_blank_input"
    EXTRACT_ALL_REPRESENTATIONS = "extract_all_representations"
    NORMALIZE_TEXT_PRESERVING_SEMANTICS = "normalize_text_preserving_semantics"
    NORMALIZE_SMART_QUOTES = "normalize_smart_quotes"
    STRIP_FENCES = "strip_fences"
    DEDENT_CANDIDATES = "dedent_candidates"
    SPLIT_ON_NAME_GUARD = "split_on_name_guard"
    REPAIR_IMPORT_LINES = "repair_import_lines"
    INFER_MISSING_IMPORTS = "infer_missing_imports"
    DEDUPE_IMPORTS = "dedupe_imports"
    ADD_LAST_RETURN_SALVAGE = "add_last_return_salvage"
    DROP_BLANK_CANDIDATES = "drop_blank_candidates"
    DEDUPE_CANDIDATES = "dedupe_candidates"
    INSPECT_CANDIDATES = "inspect_candidates"
    FILTER_PLAIN_LITERAL = "filter_plain_literal"
    FILTER_CODE_REPR = "filter_code_repr"
    FILTER_COMPILABLE = "filter_compilable"
    FILTER_TOP_LEVEL_FUNCTIONS = "filter_top_level_functions"
    MATERIALIZE_CANDIDATE_SET = "materialize_candidate_set"


__all__ = ["StepName"]
