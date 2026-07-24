"""HumanEval parser profiles as selection policy over one pipeline.

A parser profile is a registered coordinate plus a candidate-selection
policy — never a second parsing implementation. Every profile routes the
raw submission through the registered ``humaneval-function-candidates``
preprocessing definition; the definition is exhaustive and returns every
compilable candidate carrying a top-level function, in conservative order.
A profile then decides which single candidate a scoring caller receives.

``CodeExtractionResult`` is the boundary shape scoring consumes. The full
audit log of how a submission was parsed lives on the preprocessing
``Trace`` (``values``, ``step_facts``, and each candidate's
``CandidateLineage``), which the result carries as ``trace``.
"""

from __future__ import annotations

import ast
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    StrictStr,
)

from dr_code.models import FrozenModel
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    BoundPreprocessingRunner,
    bind_preprocessing,
)
from dr_code.preprocessing.extraction import RESPONSE_REPRESENTATION_OPERATION
from dr_code.trace import (
    OUTPUT_KEY,
    CandidateLineage,
    CodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    is_absent,
)

BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID = "humaneval-best-effort"
STRICT_FIELD_MARKER_PARSER_PROFILE_ID = "humaneval-field-marker"
BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION = "0"
STRICT_FIELD_MARKER_PARSER_PROFILE_VERSION = "0"
FIELD_MARKER_NAME = "code"

#: Response representation the field-marker profile admits. Extraction emits
#: it under this exact name; the strict profile keeps only candidates whose
#: lineage starts there.
FIELD_MARKER_REPRESENTATION: Final = "field_marker_code"

EMPTY_SUBMISSION_ERROR: Final = "empty raw submission"
NO_FIELD_MARKER_ERROR: Final = (
    f"missing field marker for {FIELD_MARKER_NAME!r}"
)


class CandidateSelection(StrEnum):
    """How a profile narrows the pipeline's candidates to at most one."""

    #: Take the pipeline's first candidate — its most conservative reading.
    FIRST = "first"
    #: Take the first candidate extracted from the ``code`` field marker.
    FIRST_FIELD_MARKER = "first_field_marker"


class CodeParserProfile(FrozenModel):
    """Registered parser coordinate and its candidate-selection policy.

    ``selection`` defaults to the most conservative policy so a caller can
    name a coordinate without restating its policy; the registered profile
    is still authoritative, since ``extract_code_with_profile`` compares the
    whole object against the registry before running.
    """

    profile_id: StrictStr
    version: StrictStr
    selection: CandidateSelection = CandidateSelection.FIRST


class CodeExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    _parsed_candidate: ast.Module | None = PrivateAttr(default=None)

    raw_submission: StrictStr | None
    extracted_code: StrictStr | None
    candidate_count: StrictInt
    selected_candidate_index: StrictInt | None = None
    extraction_error: StrictStr | None = None
    profile: CodeParserProfile
    trace: Trace
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.extracted_code is not None

    @property
    def parsed_candidate(self) -> ast.Module | None:
        return self._parsed_candidate


BEST_EFFORT_HUMANEVAL_PARSER_PROFILE = CodeParserProfile(
    profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    version=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION,
    selection=CandidateSelection.FIRST,
)
STRICT_FIELD_MARKER_PARSER_PROFILE = CodeParserProfile(
    profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    version=STRICT_FIELD_MARKER_PARSER_PROFILE_VERSION,
    selection=CandidateSelection.FIRST_FIELD_MARKER,
)

_PARSER_PROFILES = MappingProxyType(
    {
        (profile.profile_id, profile.version): profile
        for profile in (
            BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
            STRICT_FIELD_MARKER_PARSER_PROFILE,
        )
    }
)


def resolve_parser_profile(
    *,
    parser_profile_id: str,
    parser_version: str,
) -> CodeParserProfile:
    profile = _PARSER_PROFILES.get((parser_profile_id, parser_version))
    if profile is None:
        raise ValueError(
            f"unsupported parser profile: {parser_profile_id}@{parser_version}"
        )
    return profile


def _registered_parser_profile(
    profile: CodeParserProfile,
) -> CodeParserProfile:
    registered = resolve_parser_profile(
        parser_profile_id=profile.profile_id,
        parser_version=profile.version,
    )
    if profile != registered:
        raise ValueError(
            "parser profile does not match its registered coordinate: "
            f"{profile.profile_id}@{profile.version}"
        )
    return registered


@lru_cache(maxsize=1)
def _pipeline() -> BoundPreprocessingRunner:
    """Bind the one registered definition once for every profile."""
    return bind_preprocessing(HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION)


def _from_field_marker(lineage: CandidateLineage) -> bool:
    """Whether any origin of this candidate starts at the ``code`` marker."""
    return any(
        origin.path[0].kind == RESPONSE_REPRESENTATION_OPERATION
        and origin.path[0].details.get("name") == FIELD_MARKER_REPRESENTATION
        for origin in lineage.origins
    )


def _select_candidate(
    candidates: CodeCandidateSetArtifact,
    *,
    selection: CandidateSelection,
) -> int | None:
    """Apply the profile's selection policy to the pipeline's output.

    The pipeline emits candidates in conservative-first order, so every
    policy here is a first-match scan; a profile narrows which candidates
    are admissible, never how they were parsed.
    """
    for index, lineage in enumerate(candidates.lineage):
        if selection is CandidateSelection.FIRST:
            return index
        if selection is CandidateSelection.FIRST_FIELD_MARKER:
            if _from_field_marker(lineage):
                return index
    return None


def _selection_error(
    *,
    raw_submission: str,
    selection: CandidateSelection,
    trace: Trace,
) -> str:
    """Name why a completed pipeline yielded no candidate for this profile."""
    if not raw_submission.strip():
        return EMPTY_SUBMISSION_ERROR
    output = trace.value(OUTPUT_KEY)
    if is_absent(output):
        return output.cause
    if selection is CandidateSelection.FIRST_FIELD_MARKER:
        return NO_FIELD_MARKER_ERROR
    return "no compilable extracted candidate"


def extract_code_with_profile(
    raw_submission: str,
    *,
    profile: CodeParserProfile,
) -> CodeExtractionResult:
    """Extract one candidate under an exact registered parser profile."""
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    profile = _registered_parser_profile(profile)
    trace = _pipeline().run(TextArtifact(text=raw_submission))
    output = trace.value(OUTPUT_KEY)

    if is_absent(output):
        return _extraction_result(
            raw_submission=raw_submission,
            profile=profile,
            trace=trace,
            candidate_count=0,
            selected_index=None,
            error=_selection_error(
                raw_submission=raw_submission,
                selection=profile.selection,
                trace=trace,
            ),
        )

    assert isinstance(output, CodeCandidateSetArtifact)
    selected_index = _select_candidate(output, selection=profile.selection)
    if selected_index is None:
        return _extraction_result(
            raw_submission=raw_submission,
            profile=profile,
            trace=trace,
            candidate_count=len(output.candidates),
            selected_index=None,
            error=_selection_error(
                raw_submission=raw_submission,
                selection=profile.selection,
                trace=trace,
            ),
        )

    selected = output.candidates[selected_index]
    result = _extraction_result(
        raw_submission=raw_submission,
        profile=profile,
        trace=trace,
        candidate_count=len(output.candidates),
        selected_index=selected_index,
        extracted_code=selected,
    )
    # The pipeline already filtered to compilable candidates, so this parse
    # only rebuilds the tree the evaluation harness reuses.
    result._parsed_candidate = ast.parse(selected)
    return result


def _extraction_result(
    *,
    raw_submission: str,
    profile: CodeParserProfile,
    trace: Trace,
    candidate_count: int,
    selected_index: int | None,
    extracted_code: str | None = None,
    error: str | None = None,
) -> CodeExtractionResult:
    return CodeExtractionResult(
        raw_submission=raw_submission,
        extracted_code=extracted_code,
        candidate_count=candidate_count,
        selected_candidate_index=selected_index,
        extraction_error=error,
        profile=profile,
        trace=trace,
        metadata={
            "candidate_count": candidate_count,
            "selected_candidate_index": selected_index,
        },
    )


__all__ = [
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE",
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID",
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_VERSION",
    "EMPTY_SUBMISSION_ERROR",
    "FIELD_MARKER_NAME",
    "FIELD_MARKER_REPRESENTATION",
    "NO_FIELD_MARKER_ERROR",
    "STRICT_FIELD_MARKER_PARSER_PROFILE",
    "STRICT_FIELD_MARKER_PARSER_PROFILE_ID",
    "STRICT_FIELD_MARKER_PARSER_PROFILE_VERSION",
    "CandidateSelection",
    "CodeExtractionResult",
    "CodeParserProfile",
    "extract_code_with_profile",
    "resolve_parser_profile",
]
