"""Render the parser-emitted extraction trace."""

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    PARSER_PROFILE_VERSION,
    ExtractionTrace,
    extract_code_with_profile,
    resolve_parser_profile,
)


def explain_extraction(
    text: str,
    *,
    profile_id: str = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    parser_version: str = PARSER_PROFILE_VERSION,
) -> ExtractionTrace:
    profile = resolve_parser_profile(
        parser_profile_id=profile_id,
        parser_version=parser_version,
    )
    return extract_code_with_profile(text, profile=profile).trace
