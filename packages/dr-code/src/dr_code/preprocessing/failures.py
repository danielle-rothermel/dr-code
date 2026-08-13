from __future__ import annotations

from enum import StrEnum, verify, UNIQUE


@verify(UNIQUE)
class PreprocessingFailureCode(StrEnum):
    # Never build payloads by iterating this closed vocabulary.

    BLANK_INPUT = "blank_input"
    NO_CANDIDATES_EXTRACTED = "no_candidates_extracted"
    NO_CANDIDATE_SURVIVED_FILTERING = "no_candidate_survived_filtering"


__all__ = ["PreprocessingFailureCode"]
