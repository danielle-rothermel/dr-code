"""The closed failure vocabulary preprocessing steps raise.

Preprocessing owns these codes; the trace layer stores whichever string a
producer stamps into ``Absent.failure_code`` without interpreting it. A
step raises ``StepFailedError`` with a member of this enum, and the runner
copies its value onto the ``Absent`` it records.

Never build a payload by iterating this enum: the set of members is a
closed vocabulary, not an ordered list, and its iteration order is not part
of any persisted format. Reference members individually by name.
"""

from __future__ import annotations

from enum import StrEnum, verify, UNIQUE


@verify(UNIQUE)
class PreprocessingFailureCode(StrEnum):
    """Every failure kind a preprocessing step may raise."""

    #: The input text is empty or whitespace-only, so there is nothing to
    #: extract from. Raised before any extraction is attempted.
    BLANK_INPUT = "blank_input"
    #: No supported representation yielded a code-like candidate. The
    #: registered exhaustive definition preempts this code: it guards with
    #: ``reject_blank_input`` first, and the raw-response reading turns any
    #: non-blank input into at least one candidate. It stays reachable for
    #: externally bound definitions that omit the blank guard.
    NO_CANDIDATES_EXTRACTED = "no_candidates_extracted"
    #: Every extracted candidate was dropped by the structural filters.
    NO_CANDIDATE_SURVIVED_FILTERING = "no_candidate_survived_filtering"


__all__ = ["PreprocessingFailureCode"]
