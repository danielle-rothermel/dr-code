"""Stage-by-stage explanation of code extraction.

Mirrors the selection walk in `extract_best_effort_code` /
`extract_strict_field_marker_code` while recording why each candidate
was rejected, so the parser playground can render a candidate tree with
a winner rationale. The canonical outcome always comes from
`extract_code_with_profile`; the per-candidate annotations replay the
same helper checks in the same order.
"""

from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.code_extraction import apply_cleaning
from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    FIELD_MARKER_NAME,
    PARSER_PROFILE_VERSION,
    STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    CodeExtractionResult,
    CodeParserProfile,
    extract_code_with_profile,
    field_marker_value,
    is_code_repr_assignment,
    is_plain_literal_module,
    resolve_parser_profile,
)

PLAIN_LITERAL_REJECTION = "plain literal modules are not valid HumanEval code"
CODE_REPR_REJECTION = "code repr assignments are not valid HumanEval code"


class ExplainStage(StrEnum):
    UNWRAP = "unwrap"
    CANDIDATES = "candidates"
    SELECTION = "selection"
    RESULT = "result"


ALL_EXPLAIN_STAGES = frozenset(ExplainStage)


class CandidateStatus(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"
    NOT_REACHED = "not_reached"


class UnwrapExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unwrapped_text: StrictStr | None
    method: StrictStr | None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class CandidateExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: StrictInt
    source: StrictStr
    status: CandidateStatus
    compile_ok: StrictBool
    rejection_reason: StrictStr | None = None


class SelectionExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_index: StrictInt | None
    method: StrictStr | None
    rationale: StrictStr


class ExtractionExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: CodeParserProfile
    stages: list[ExplainStage]
    unwrap: UnwrapExplanation | None = None
    candidates: list[CandidateExplanation] | None = None
    selection: SelectionExplanation | None = None
    result: CodeExtractionResult | None = None


def candidate_rejection_reason(source: str) -> str | None:
    validation = validate_python_source(source)
    if not validation.compile_ok:
        return validation.compile_error or "candidate does not compile"
    if is_plain_literal_module(source):
        return PLAIN_LITERAL_REJECTION
    if is_code_repr_assignment(source):
        return CODE_REPR_REJECTION
    return None


def annotate_candidates(
    sources: list[str],
    *,
    selected_index: int | None,
) -> list[CandidateExplanation]:
    annotated: list[CandidateExplanation] = []
    for index, source in enumerate(sources):
        compile_ok = validate_python_source(source).compile_ok
        if selected_index is not None and index > selected_index:
            status = CandidateStatus.NOT_REACHED
            reason = None
        elif index == selected_index:
            status = CandidateStatus.SELECTED
            reason = None
        else:
            status = CandidateStatus.REJECTED
            reason = candidate_rejection_reason(source)
        annotated.append(
            CandidateExplanation(
                index=index,
                source=source,
                status=status,
                compile_ok=compile_ok,
                rejection_reason=reason,
            )
        )
    return annotated


def selection_rationale(result: CodeExtractionResult) -> str:
    if result.succeeded:
        method = (
            result.extraction_method.value
            if result.extraction_method is not None
            else "unknown"
        )
        return (
            f"candidate {result.selected_candidate_index} selected via "
            f"{method}: first compilable candidate passing the plain-literal "
            "and code-repr checks"
        )
    reason = result.extraction_error or "extraction failed"
    if result.compile_error:
        return f"{reason} ({result.compile_error})"
    return reason


def explain_unwrap(
    text: str,
    *,
    profile: CodeParserProfile,
) -> tuple[UnwrapExplanation, str | None]:
    if profile.profile_id == STRICT_FIELD_MARKER_PARSER_PROFILE_ID:
        marker_value = field_marker_value(
            text,
            field_name=FIELD_MARKER_NAME,
        )
        return (
            UnwrapExplanation(
                unwrapped_text=marker_value,
                method="field_marker" if marker_value is not None else None,
                metadata={"field_name": FIELD_MARKER_NAME},
            ),
            marker_value,
        )
    return (
        UnwrapExplanation(
            unwrapped_text=text,
            method=None,
            metadata={},
        ),
        text,
    )


def candidate_sources(
    unwrapped: str | None,
    *,
    profile: CodeParserProfile,
) -> list[str]:
    if unwrapped is None or not unwrapped.strip():
        return []
    if profile.profile_id == STRICT_FIELD_MARKER_PARSER_PROFILE_ID:
        return [unwrapped.strip()]
    return apply_cleaning(unwrapped, apply_dedent=True)


def explain_extraction(
    text: str,
    *,
    profile_id: str = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    parser_version: str = PARSER_PROFILE_VERSION,
    code_field: str = FIELD_MARKER_NAME,
    stages: frozenset[ExplainStage] | None = None,
) -> ExtractionExplanation:
    _ = code_field
    profile = resolve_parser_profile(
        parser_profile_id=profile_id,
        parser_version=parser_version,
    )
    requested = ALL_EXPLAIN_STAGES if stages is None else stages
    result = extract_code_with_profile(text, profile=profile)

    unwrap_info, unwrapped = explain_unwrap(text, profile=profile)
    sources = candidate_sources(unwrapped, profile=profile)
    candidates = annotate_candidates(
        sources,
        selected_index=result.selected_candidate_index,
    )
    selection = SelectionExplanation(
        selected_index=result.selected_candidate_index,
        method=(
            result.extraction_method.value
            if result.extraction_method is not None
            else None
        ),
        rationale=selection_rationale(result),
    )

    return ExtractionExplanation(
        profile=profile,
        stages=sorted(requested),
        unwrap=unwrap_info if ExplainStage.UNWRAP in requested else None,
        candidates=(
            candidates if ExplainStage.CANDIDATES in requested else None
        ),
        selection=selection if ExplainStage.SELECTION in requested else None,
        result=result if ExplainStage.RESULT in requested else None,
    )
