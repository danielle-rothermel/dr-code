from __future__ import annotations

import ast
import re
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    StrictStr,
)

from dr_code.code_analysis import validate_python_source_with_ast
from dr_code.humaneval.code_extraction import (
    ExtractionTraceNode,
    TraceCheckVerdict,
    TraceNodeKind,
    apply_cleaning_with_trace,
)

BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID = "humaneval-best-effort"
STRICT_FIELD_MARKER_PARSER_PROFILE_ID = "humaneval-field-marker"
LEGACY_PARSER_PROFILE_VERSION = "v1"
PARSER_PROFILE_VERSION = "v2"
SUPPORTED_PARSER_PROFILE_VERSIONS = {
    LEGACY_PARSER_PROFILE_VERSION,
    PARSER_PROFILE_VERSION,
}
FIELD_MARKER_NAME = "code"
FIELD_MARKER_RE = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)


class ExtractionMethod(StrEnum):
    FENCED_CODE = "fenced_code"
    CLEANED_CANDIDATE = "cleaned_candidate"
    BARE_PYTHON = "bare_python"
    FIELD_MARKER = "field_marker"


class CodeParserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: StrictStr
    version: StrictStr


class CandidateStatus(StrEnum):
    SELECTED = "selected"
    REJECTED = "rejected"
    NOT_REACHED = "not_reached"


class CandidateSelectionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: StrictInt
    source: StrictStr
    status: CandidateStatus
    compile_ok: StrictBool | None = None
    rejection_reason: StrictStr | None = None
    checks: list[ExtractionTraceNode] = Field(default_factory=list)


class ExtractionTrace(BaseModel):
    """How one submission was parsed, split into contract vs. diagnostic.

    Product contract (stable; downstream consumers may depend on them):
        `profile`, `extraction_method`, `selected_candidate_index`,
        `extraction_error`. These name the *outcome* of parsing (which
        profile ran, which method won, why nothing was selected) and are the
        fields callers should assert against.

    Diagnostic / internal (may change between versions without notice):
        `roots` and each `CandidateSelectionTrace`'s per-node structure,
        `rationale`. The node `name`s mirror the pipeline's transform and
        check steps one-to-one, so they shift whenever the pipeline is
        refactored. Treat them as a human-debuggable audit log, not an API:
        do not pin node names, node counts, or child ordering.
    """

    model_config = ConfigDict(extra="forbid")

    profile: CodeParserProfile
    roots: list[ExtractionTraceNode]
    candidates: list[CandidateSelectionTrace]
    selected_candidate_index: StrictInt | None = None
    extraction_method: ExtractionMethod | None = None
    rationale: StrictStr
    extraction_error: StrictStr | None = None


class CodeExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    _parsed_candidate: ast.Module | None = PrivateAttr(default=None)

    raw_submission: StrictStr | None
    extracted_code: StrictStr | None
    extraction_method: ExtractionMethod | None
    candidate_count: StrictInt
    selected_candidate_index: StrictInt | None = None
    compile_ok: bool
    compile_error: StrictStr | None = None
    extraction_error: StrictStr | None = None
    trace: ExtractionTrace
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.extracted_code is not None

    @property
    def parsed_candidate(self) -> ast.Module | None:
        return self._parsed_candidate


BEST_EFFORT_HUMANEVAL_PARSER_PROFILE = CodeParserProfile(
    profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    version=PARSER_PROFILE_VERSION,
)
STRICT_FIELD_MARKER_PARSER_PROFILE = CodeParserProfile(
    profile_id=STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    version=PARSER_PROFILE_VERSION,
)


def resolve_parser_profile(
    *,
    parser_profile_id: str,
    parser_version: str,
) -> CodeParserProfile:
    if parser_version not in SUPPORTED_PARSER_PROFILE_VERSIONS:
        raise ValueError(
            f"unsupported parser profile version: {parser_version}"
        )
    if parser_profile_id not in {
        BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        STRICT_FIELD_MARKER_PARSER_PROFILE_ID,
    }:
        raise ValueError(f"unsupported parser profile id: {parser_profile_id}")
    return CodeParserProfile(
        profile_id=parser_profile_id,
        version=parser_version,
    )


def extract_code_with_profile(
    raw_submission: str,
    *,
    profile: CodeParserProfile,
) -> CodeExtractionResult:
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    if profile.profile_id == BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID:
        return extract_best_effort_code(raw_submission, profile=profile)
    if profile.profile_id == STRICT_FIELD_MARKER_PARSER_PROFILE_ID:
        return extract_strict_field_marker_code(
            raw_submission,
            profile=profile,
        )
    raise ValueError(f"unsupported parser profile id: {profile.profile_id}")


def extract_best_effort_code(
    raw_submission: str,
    *,
    profile: CodeParserProfile = BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
) -> CodeExtractionResult:
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    cleaning = apply_cleaning_with_trace(
        raw_submission,
        apply_dedent=True,
        unescape_fallback=profile.version != LEGACY_PARSER_PROFILE_VERSION,
    )
    if not raw_submission.strip():
        trace = build_extraction_trace(
            profile=profile,
            roots=cleaning.roots,
            candidates=[],
            rationale="empty raw submission",
            extraction_error="empty raw submission",
        )
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=0,
            error="empty raw submission",
            trace=trace,
        )

    candidates = cleaning.candidates
    if not candidates:
        trace = build_extraction_trace(
            profile=profile,
            roots=cleaning.roots,
            candidates=[],
            rationale="no code candidates extracted",
            extraction_error="no code candidates extracted",
        )
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=0,
            error="no code candidates extracted",
            trace=trace,
        )

    first_compile_error: str | None = None
    selected_index: int | None = None
    selected_candidate: str | None = None
    selected_parsed_candidate: ast.Module | None = None
    selected_extraction_method: ExtractionMethod | None = None
    candidate_traces: list[CandidateSelectionTrace] = []
    for index, candidate in enumerate(candidates):
        if selected_index is not None:
            candidate_traces.append(
                CandidateSelectionTrace(
                    index=index,
                    source=candidate,
                    status=CandidateStatus.NOT_REACHED,
                )
            )
            continue

        candidate_trace, parsed_candidate = candidate_selection(
            candidate,
            index=index,
        )
        candidate_traces.append(candidate_trace)
        if candidate_trace.status is CandidateStatus.REJECTED:
            first_compile_error = (
                first_compile_error or candidate_trace.rejection_reason
            )
            continue

        selected_index = index
        selected_candidate = candidate
        selected_parsed_candidate = parsed_candidate
        selected_extraction_method = selected_method(
            raw_submission=raw_submission,
            candidate=candidate,
        )

    if selected_index is not None and selected_candidate is not None:
        method = (
            selected_extraction_method or ExtractionMethod.CLEANED_CANDIDATE
        )
        rationale = selection_rationale(
            selected_candidate_index=selected_index,
            extraction_method=method,
        )
        trace = build_extraction_trace(
            profile=profile,
            roots=cleaning.roots,
            candidates=candidate_traces,
            selected_candidate_index=selected_index,
            extraction_method=method,
            rationale=rationale,
        )
        result = CodeExtractionResult(
            raw_submission=raw_submission,
            extracted_code=selected_candidate,
            extraction_method=method,
            candidate_count=len(candidates),
            selected_candidate_index=selected_index,
            compile_ok=True,
            compile_error=None,
            trace=trace,
            metadata={
                "candidate_count": len(candidates),
                "selected_candidate_index": selected_index,
            },
        )
        result._parsed_candidate = selected_parsed_candidate
        return result

    trace = build_extraction_trace(
        profile=profile,
        roots=cleaning.roots,
        candidates=candidate_traces,
        rationale=failure_rationale(
            error="no compilable extracted candidate",
            compile_error=first_compile_error,
        ),
        extraction_error="no compilable extracted candidate",
    )
    return extraction_failure(
        raw_submission=raw_submission,
        candidate_count=len(candidates),
        error="no compilable extracted candidate",
        compile_error=first_compile_error,
        trace=trace,
        metadata={
            "candidate_count": len(candidates),
        },
    )


def extract_strict_field_marker_code(
    raw_submission: str,
    *,
    profile: CodeParserProfile = STRICT_FIELD_MARKER_PARSER_PROFILE,
) -> CodeExtractionResult:
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    field_value = field_marker_value(
        raw_submission,
        field_name=FIELD_MARKER_NAME,
    )
    roots = strict_field_marker_roots(
        raw_submission=raw_submission,
        field_value=field_value,
    )
    if field_value is None:
        error = f"missing field marker for {FIELD_MARKER_NAME!r}"
        trace = build_extraction_trace(
            profile=profile,
            roots=roots,
            candidates=[],
            rationale=error,
            extraction_error=error,
        )
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=0,
            error=error,
            trace=trace,
        )
    candidate = field_value.strip()
    if not candidate:
        error = "empty field-marker code"
        trace = build_extraction_trace(
            profile=profile,
            roots=roots,
            candidates=[],
            rationale=error,
            extraction_error=error,
        )
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=1,
            error=error,
            trace=trace,
        )
    candidate_trace, parsed_candidate = candidate_selection(
        candidate,
        index=0,
        include_code_repr_check=False,
    )
    if candidate_trace.status is CandidateStatus.REJECTED:
        error = (
            "field-marker code is not compilable"
            if candidate_trace.compile_ok is False
            else candidate_trace.rejection_reason
            or "field-marker code is rejected"
        )
        trace = build_extraction_trace(
            profile=profile,
            roots=roots,
            candidates=[candidate_trace],
            rationale=failure_rationale(
                error=error,
                compile_error=(
                    candidate_trace.rejection_reason
                    if candidate_trace.compile_ok is False
                    else None
                ),
            ),
            extraction_error=error,
        )
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=1,
            error=error,
            compile_error=(
                candidate_trace.rejection_reason
                if candidate_trace.compile_ok is False
                else None
            ),
            trace=trace,
        )
    trace = build_extraction_trace(
        profile=profile,
        roots=roots,
        candidates=[candidate_trace],
        selected_candidate_index=0,
        extraction_method=ExtractionMethod.FIELD_MARKER,
        rationale=selection_rationale(
            selected_candidate_index=0,
            extraction_method=ExtractionMethod.FIELD_MARKER,
        ),
    )
    result = CodeExtractionResult(
        raw_submission=raw_submission,
        extracted_code=candidate,
        extraction_method=ExtractionMethod.FIELD_MARKER,
        candidate_count=1,
        selected_candidate_index=0,
        compile_ok=True,
        trace=trace,
        metadata={
            "candidate_count": 1,
            "selected_candidate_index": 0,
            "field_name": FIELD_MARKER_NAME,
        },
    )
    result._parsed_candidate = parsed_candidate
    return result


def build_extraction_trace(
    *,
    profile: CodeParserProfile,
    roots: list[ExtractionTraceNode],
    candidates: list[CandidateSelectionTrace],
    rationale: str,
    selected_candidate_index: int | None = None,
    extraction_method: ExtractionMethod | None = None,
    extraction_error: str | None = None,
) -> ExtractionTrace:
    return ExtractionTrace(
        profile=profile,
        roots=roots,
        candidates=candidates,
        selected_candidate_index=selected_candidate_index,
        extraction_method=extraction_method,
        rationale=rationale,
        extraction_error=extraction_error,
    )


def trace_candidate_selection(
    candidate: str,
    *,
    index: int,
    include_code_repr_check: bool = True,
) -> CandidateSelectionTrace:
    trace, _ = candidate_selection(
        candidate,
        index=index,
        include_code_repr_check=include_code_repr_check,
    )
    return trace


def candidate_selection(
    candidate: str,
    *,
    index: int,
    include_code_repr_check: bool = True,
) -> tuple[CandidateSelectionTrace, ast.Module | None]:
    checks: list[ExtractionTraceNode] = []
    validated = validate_python_source_with_ast(candidate)
    validation = validated.validation
    parsed_module = validated.tree
    if not validation.compile_ok:
        reason = validation.compile_error or "candidate does not compile"
        checks.append(check_node("compile_validation", False, reason=reason))
        return (
            CandidateSelectionTrace(
                index=index,
                source=candidate,
                status=CandidateStatus.REJECTED,
                compile_ok=False,
                rejection_reason=reason,
                checks=checks,
            ),
            parsed_module,
        )
    checks.append(check_node("compile_validation", True))

    if is_plain_literal_module(candidate, parsed_module=parsed_module):
        reason = "plain literal modules are not valid HumanEval code"
        checks.append(check_node("plain_literal_module", False, reason=reason))
        return (
            CandidateSelectionTrace(
                index=index,
                source=candidate,
                status=CandidateStatus.REJECTED,
                compile_ok=True,
                rejection_reason=reason,
                checks=checks,
            ),
            parsed_module,
        )
    checks.append(check_node("plain_literal_module", True))

    if include_code_repr_check:
        if is_code_repr_assignment(candidate, parsed_module=parsed_module):
            reason = "code repr assignments are not valid HumanEval code"
            checks.append(
                check_node("code_repr_assignment", False, reason=reason)
            )
            return (
                CandidateSelectionTrace(
                    index=index,
                    source=candidate,
                    status=CandidateStatus.REJECTED,
                    compile_ok=True,
                    rejection_reason=reason,
                    checks=checks,
                ),
                parsed_module,
            )
        checks.append(check_node("code_repr_assignment", True))

    return (
        CandidateSelectionTrace(
            index=index,
            source=candidate,
            status=CandidateStatus.SELECTED,
            compile_ok=True,
            checks=checks,
        ),
        parsed_module,
    )


def check_node(
    check_name: str,
    passed: bool,
    *,
    reason: str | None = None,
) -> ExtractionTraceNode:
    return ExtractionTraceNode(
        kind=TraceNodeKind.CHECK,
        name=check_name,
        check_name=check_name,
        verdict=TraceCheckVerdict.PASS if passed else TraceCheckVerdict.FAIL,
        reason=reason,
    )


def selection_rationale(
    *,
    selected_candidate_index: int,
    extraction_method: ExtractionMethod,
) -> str:
    return (
        f"candidate {selected_candidate_index} selected via "
        f"{extraction_method.value}: first candidate passing parser checks"
    )


def failure_rationale(
    *,
    error: str,
    compile_error: str | None,
) -> str:
    if compile_error:
        return f"{error} ({compile_error})"
    return error


def strict_field_marker_roots(
    *,
    raw_submission: str,
    field_value: str | None,
) -> list[ExtractionTraceNode]:
    marker_present = field_value is not None
    marker_node = check_node(
        "field_marker_present",
        marker_present,
        reason=(
            None
            if marker_present
            else f"missing field marker for {FIELD_MARKER_NAME!r}"
        ),
    )
    if field_value is None:
        return [marker_node]

    extract_node = ExtractionTraceNode(
        kind=TraceNodeKind.TRANSFORM,
        name="field_marker_extract",
        before_text=raw_submission,
        after_text=field_value,
    )
    strip_node = ExtractionTraceNode(
        kind=TraceNodeKind.TRANSFORM,
        name="field_marker_strip",
        before_text=field_value,
        after_text=field_value.strip(),
    )
    extract_node.children = [strip_node]
    marker_node.children = [extract_node]
    return [marker_node]


def field_marker_value(raw_submission: str, *, field_name: str) -> str | None:
    matches = list(FIELD_MARKER_RE.finditer(raw_submission))
    for index, match in enumerate(matches):
        if match.group("field") != field_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return raw_submission[start:end]
    return None


def selected_method(
    *,
    raw_submission: str,
    candidate: str,
) -> ExtractionMethod:
    if "```" in raw_submission or "~~~" in raw_submission:
        return ExtractionMethod.FENCED_CODE
    if raw_submission.strip() == candidate.strip():
        return ExtractionMethod.BARE_PYTHON
    return ExtractionMethod.CLEANED_CANDIDATE


def is_plain_literal_module(
    source: str,
    *,
    parsed_module: ast.Module | None = None,
) -> bool:
    if parsed_module is None:
        try:
            parsed_module = ast.parse(source)
        except (SyntaxError, ValueError):
            return False
    tree = parsed_module
    if len(tree.body) != 1:
        return False
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr):
        return False
    return isinstance(stmt.value, ast.Dict | ast.List | ast.Set | ast.Tuple)


def is_code_repr_assignment(
    source: str,
    *,
    parsed_module: ast.Module | None = None,
) -> bool:
    if parsed_module is None:
        try:
            parsed_module = ast.parse(source)
        except (SyntaxError, ValueError):
            return False
    tree = parsed_module
    if len(tree.body) != 1:
        return False
    statement = tree.body[0]
    if not isinstance(statement, ast.Assign):
        return False
    if len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if not isinstance(target, ast.Name) or target.id != FIELD_MARKER_NAME:
        return False
    return isinstance(statement.value, ast.Constant) and isinstance(
        statement.value.value,
        str,
    )


def extraction_failure(
    *,
    raw_submission: str | None,
    candidate_count: int,
    error: str,
    trace: ExtractionTrace,
    compile_error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CodeExtractionResult:
    return CodeExtractionResult(
        raw_submission=raw_submission,
        extracted_code=None,
        extraction_method=None,
        candidate_count=candidate_count,
        selected_candidate_index=None,
        compile_ok=False,
        compile_error=compile_error,
        extraction_error=error,
        trace=trace,
        metadata={
            **(metadata or {}),
            "candidate_count": candidate_count,
        },
    )
