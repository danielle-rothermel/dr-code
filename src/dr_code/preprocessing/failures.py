"""Stable machine-readable preprocessing failure codes."""

from __future__ import annotations

from enum import StrEnum


class PreprocessingFailureCode(StrEnum):
    """Closed failure vocabulary emitted by preprocessing steps."""

    CODE_REPR_ONLY = "code_repr_only"
    DECODER_OUTPUT_BLANK = "decoder_output_blank"
    DECODER_OUTPUT_INVALID = "decoder_output_invalid"
    NO_CANDIDATES_TO_IDENTIFY = "no_candidates_to_identify"
    NO_CANDIDATES_TO_RETURN = "no_candidates_to_return"
    NO_CODE_CANDIDATES = "no_code_candidates"
    NO_COMPILABLE_CANDIDATE = "no_compilable_candidate"
    NO_NONBLANK_CLEANED_CANDIDATE = "no_nonblank_cleaned_candidate"
    NO_TOP_LEVEL_FUNCTION_CANDIDATE = "no_top_level_function_candidate"
    PLAIN_LITERAL_ONLY = "plain_literal_only"


__all__ = ["PreprocessingFailureCode"]
