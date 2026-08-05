"""Names for preprocessing pipeline steps."""

from __future__ import annotations

from enum import StrEnum


class StepName(StrEnum):
    """All preprocessing pipeline steps.

    Atomic text steps compose the ``normalize_text`` behavior. A guard step
    rejects blank input before extraction. Candidate-generation steps emit
    ``CodeCandidateSetArtifact`` from text, additively across every
    supported representation. Elementwise steps map over a candidate set.
    Set-shaping steps drop blanks and merge duplicates. Inspection pairs
    every candidate with the structural facts of its exact source, and the
    filters read those stored facts rather than reparsing.
    """

    # atomic text steps — the constituents of normalize_text
    NORMALIZE_LINE_ENDINGS = "normalize_line_endings"
    NORMALIZE_UNICODE = "normalize_unicode"  # NFKC
    EXPAND_TABS = "expand_tabs"  # settings: tab_width
    STRIP_TRAILING_WHITESPACE = "strip_trailing_whitespace"
    COLLAPSE_BLANK_RUNS = "collapse_blank_runs"
    TRIM_OUTER_BLANKS = "trim_outer_blanks"
    # input guard
    REJECT_BLANK_INPUT = "reject_blank_input"
    # candidate generation — additive across representations
    EXTRACT_ALL_REPRESENTATIONS = "extract_all_representations"
    # elementwise candidate transforms
    NORMALIZE_SMART_QUOTES = "normalize_smart_quotes"
    STRIP_FENCES = "strip_fences"
    DEDENT_CANDIDATES = "dedent_candidates"
    SPLIT_ON_NAME_GUARD = "split_on_name_guard"  # drop_if_name
    # import handling — infer_necessary_imports unbundled
    REPAIR_IMPORT_LINES = "repair_import_lines"
    INFER_MISSING_IMPORTS = "infer_missing_imports"
    DEDUPE_IMPORTS = "dedupe_imports"
    # additive salvage and set shaping
    ADD_LAST_RETURN_SALVAGE = "add_last_return_salvage"
    DROP_BLANK_CANDIDATES = "drop_blank_candidates"
    DEDUPE_CANDIDATES = "dedupe_candidates"
    # inspection and the filters that read it
    INSPECT_CANDIDATES = "inspect_candidates"
    FILTER_PLAIN_LITERAL = "filter_plain_literal"
    FILTER_CODE_REPR = "filter_code_repr"
    FILTER_COMPILABLE = "filter_compilable"
    FILTER_TOP_LEVEL_FUNCTIONS = "filter_top_level_functions"
    MATERIALIZE_CANDIDATE_SET = "materialize_candidate_set"


__all__ = ["StepName"]
