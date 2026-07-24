"""HumanEval preprocessing coordinates and extraction result contracts."""

from __future__ import annotations

import ast
import re

from pydantic import BaseModel, ConfigDict, StrictStr

from dr_code.trace import SerializedTrace

BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID = "humaneval-best-effort"
STRICT_FIELD_MARKER_PARSER_PROFILE_ID = "humaneval-field-marker"
PARSER_PROFILE_VERSION = "v2"
SUPPORTED_PARSER_PROFILE_VERSIONS = {PARSER_PROFILE_VERSION}
FIELD_MARKER_NAME = "code"
FIELD_MARKER_RE = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)


class CodeParserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: StrictStr
    version: StrictStr


class CodeExtractionResult(BaseModel):
    """Canonical preprocessing output translated for HumanEval scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_submission: StrictStr
    extracted_code: StrictStr | None
    extraction_error: StrictStr | None = None
    trace: SerializedTrace

    @property
    def succeeded(self) -> bool:
        return self.extracted_code is not None

    @property
    def parsed_candidate(self) -> ast.Module | None:
        if self.extracted_code is None:
            return None
        return ast.parse(self.extracted_code)


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


def field_marker_value(raw_submission: str, *, field_name: str) -> str | None:
    matches = list(FIELD_MARKER_RE.finditer(raw_submission))
    for index, match in enumerate(matches):
        if match.group("field") != field_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return raw_submission[start:end]
    return None


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
    if len(parsed_module.body) != 1:
        return False
    statement = parsed_module.body[0]
    return isinstance(statement, ast.Expr) and isinstance(
        statement.value,
        ast.Dict | ast.List | ast.Set | ast.Tuple,
    )


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
    if len(parsed_module.body) != 1:
        return False
    statement = parsed_module.body[0]
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if not isinstance(target, ast.Name) or target.id != FIELD_MARKER_NAME:
        return False
    return isinstance(statement.value, ast.Constant) and isinstance(
        statement.value.value,
        str,
    )


__all__ = [
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE",
    "BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID",
    "CodeExtractionResult",
    "CodeParserProfile",
    "FIELD_MARKER_NAME",
    "PARSER_PROFILE_VERSION",
    "STRICT_FIELD_MARKER_PARSER_PROFILE",
    "STRICT_FIELD_MARKER_PARSER_PROFILE_ID",
    "SUPPORTED_PARSER_PROFILE_VERSIONS",
    "field_marker_value",
    "is_code_repr_assignment",
    "is_plain_literal_module",
    "resolve_parser_profile",
]
