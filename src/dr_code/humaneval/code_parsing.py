from __future__ import annotations

import ast
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from dr_code.code_analysis import validate_python_source
from dr_code.humaneval.code_extraction import apply_cleaning

BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID = "humaneval-best-effort"
STRICT_FIELD_MARKER_PARSER_PROFILE_ID = "humaneval-field-marker"
PARSER_PROFILE_VERSION = "v1"
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


class CodeExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_submission: StrictStr | None
    extracted_code: StrictStr | None
    extraction_method: ExtractionMethod | None
    candidate_count: StrictInt
    selected_candidate_index: StrictInt | None = None
    compile_ok: bool
    compile_error: StrictStr | None = None
    extraction_error: StrictStr | None = None
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.extracted_code is not None


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
    if parser_version != PARSER_PROFILE_VERSION:
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
    _ = profile
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    if not raw_submission.strip():
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=0,
            error="empty raw submission",
        )

    candidates = apply_cleaning(raw_submission, apply_dedent=True)
    if not candidates:
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=0,
            error="no code candidates extracted",
        )

    first_compile_error: str | None = None
    for index, candidate in enumerate(candidates):
        validation = validate_python_source(candidate)
        if not validation.compile_ok:
            first_compile_error = (
                first_compile_error or validation.compile_error
            )
            continue
        if is_plain_literal_module(candidate):
            first_compile_error = first_compile_error or (
                "plain literal modules are not valid HumanEval code"
            )
            continue
        if is_code_repr_assignment(candidate):
            first_compile_error = first_compile_error or (
                "code repr assignments are not valid HumanEval code"
            )
            continue
        method = selected_method(
            raw_submission=raw_submission,
            candidate=candidate,
        )
        return CodeExtractionResult(
            raw_submission=raw_submission,
            extracted_code=candidate,
            extraction_method=method,
            candidate_count=len(candidates),
            selected_candidate_index=index,
            compile_ok=True,
            compile_error=None,
            metadata={
                "candidate_count": len(candidates),
                "selected_candidate_index": index,
            },
        )

    return extraction_failure(
        raw_submission=raw_submission,
        candidate_count=len(candidates),
        error="no compilable extracted candidate",
        compile_error=first_compile_error,
        metadata={
            "candidate_count": len(candidates),
        },
    )


def extract_strict_field_marker_code(
    raw_submission: str,
    *,
    profile: CodeParserProfile = STRICT_FIELD_MARKER_PARSER_PROFILE,
) -> CodeExtractionResult:
    _ = profile
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    field_value = field_marker_value(
        raw_submission,
        field_name=FIELD_MARKER_NAME,
    )
    if field_value is None:
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=0,
            error=f"missing field marker for {FIELD_MARKER_NAME!r}",
        )
    candidate = field_value.strip()
    if not candidate:
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=1,
            error="empty field-marker code",
        )
    validation = validate_python_source(candidate)
    if not validation.compile_ok:
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=1,
            error="field-marker code is not compilable",
            compile_error=validation.compile_error,
        )
    if is_plain_literal_module(candidate):
        return extraction_failure(
            raw_submission=raw_submission,
            candidate_count=1,
            error="plain literal modules are not valid HumanEval code",
        )
    return CodeExtractionResult(
        raw_submission=raw_submission,
        extracted_code=candidate,
        extraction_method=ExtractionMethod.FIELD_MARKER,
        candidate_count=1,
        selected_candidate_index=0,
        compile_ok=True,
        metadata={
            "candidate_count": 1,
            "selected_candidate_index": 0,
            "field_name": FIELD_MARKER_NAME,
        },
    )


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


def is_plain_literal_module(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    if len(tree.body) != 1:
        return False
    stmt = tree.body[0]
    if not isinstance(stmt, ast.Expr):
        return False
    return isinstance(stmt.value, ast.Dict | ast.List | ast.Set | ast.Tuple)


def is_code_repr_assignment(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
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
        metadata={
            **(metadata or {}),
            "candidate_count": candidate_count,
        },
    )
