from __future__ import annotations

import json
import re
from enum import StrEnum, verify, UNIQUE
from typing import ClassVar, Final

from dr_code.core.source.text_analysis import code_like_blocks, split_by_fences
from dr_code.core.source.text_transforms import (
    recover_escaped_python,
    strip_markdown_wrappers,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    JsonFactValue,
    TextArtifact,
)

FIELD_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)
CODE_FIELD_NAME: Final[str] = "code"
_IGNORED_JSON_NUMBER: Final[object] = object()


def _ignore_json_number(_value: str) -> object:
    return _IGNORED_JSON_NUMBER


@verify(UNIQUE)
class Representation(StrEnum):
    JSON_CODE_FIELD = "json_code_field"
    FIELD_MARKER = "field_marker"
    RAW_RESPONSE = "raw_response"
    TEXT_SEGMENTS = "text_segments"
    MARKDOWN_SEGMENTS = "markdown_segments"
    JSON_STRING_RESPONSE = "json_string_response"
    ESCAPED_PYTHON = "escaped_python"
    ESCAPED_MARKDOWN = "escaped_markdown"


def field_marker_value(text: str, *, field_name: str) -> str | None:
    matches = list(FIELD_MARKER_RE.finditer(text))
    for index, match in enumerate(matches):
        if match.group("field") != field_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return text[start:end]
    return None


def _segment_sources(blocks: list[str]) -> list[str]:
    return [block for block in code_like_blocks(blocks) if block.strip()]


def _additive_blocks(text: str) -> list[str]:
    unfenced, fenced = split_by_fences(text)
    return [*fenced, *(block for block in unfenced if block.strip())]


def _raw_response(text: str) -> list[str]:
    return [text] if text.strip() else []


def _text_segments(text: str) -> list[str]:
    return _segment_sources(_additive_blocks(text))


def _markdown_segments(text: str) -> list[str]:
    return _segment_sources(
        [strip_markdown_wrappers(block) for block in _additive_blocks(text)]
    )


def _json_string_response(text: str) -> list[str]:
    stripped = text.strip()
    if not (stripped.startswith('"') and stripped.endswith('"')):
        return []
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, str) or decoded == text:
        return []
    return _segment_sources(_additive_blocks(decoded))


def _declared_code_sources(value: str) -> list[str]:
    blocks = _additive_blocks(value)
    candidates = [
        *_segment_sources(blocks),
        *_segment_sources(
            [strip_markdown_wrappers(block) for block in blocks]
        ),
    ]
    seen = {value}
    ordered = [value]
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _json_object_code_value(text: str) -> str | None:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        decoded = json.loads(
            stripped,
            parse_float=_ignore_json_number,
            parse_int=_ignore_json_number,
            parse_constant=_ignore_json_number,
        )
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, dict):
        return None
    value = decoded.get(CODE_FIELD_NAME)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _envelope_fences(text: str) -> list[str]:
    region = field_marker_value(text, field_name=CODE_FIELD_NAME)
    return split_by_fences(text if region is None else region)[1]


def _json_code_field(text: str) -> list[str]:
    values = [
        value
        for source in (text, *_envelope_fences(text))
        if (value := _json_object_code_value(source)) is not None
    ]
    sources: list[str] = []
    for value in values:
        sources.extend(
            source
            for source in _declared_code_sources(value)
            if source not in sources
        )
    return sources


def _field_marker(text: str) -> list[str]:
    value = field_marker_value(text, field_name=CODE_FIELD_NAME)
    if value is None or not value.strip():
        return []
    return _declared_code_sources(value.strip())


def _escaped_python(text: str) -> list[str]:
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return []
    return _segment_sources(_additive_blocks(unescaped))


def _escaped_markdown(text: str) -> list[str]:
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return []
    return _segment_sources(
        [
            strip_markdown_wrappers(block)
            for block in _additive_blocks(unescaped)
        ]
    )


# Declared code fields precede inferred scrapes; every reading contributes.
_READINGS: Final = (
    (Representation.JSON_CODE_FIELD, _json_code_field),
    (Representation.FIELD_MARKER, _field_marker),
    (Representation.RAW_RESPONSE, _raw_response),
    (Representation.TEXT_SEGMENTS, _text_segments),
    (Representation.MARKDOWN_SEGMENTS, _markdown_segments),
    (Representation.JSON_STRING_RESPONSE, _json_string_response),
    (Representation.ESCAPED_PYTHON, _escaped_python),
    (Representation.ESCAPED_MARKDOWN, _escaped_markdown),
)


class ExtractAllRepresentations(Step[StepSettings]):
    NAME: ClassVar[StepName] = StepName.EXTRACT_ALL_REPRESENTATIONS
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        candidates: list[CodeCandidate] = []
        counts: dict[str, JsonFactValue] = {}
        for representation, reading in _READINGS:
            sources = reading(value.text)
            counts[representation.value] = len(sources)
            operation = ExtractionOperation(
                operation_name=representation.value
            )
            candidates.extend(
                CodeCandidate(
                    source=source,
                    origins=(
                        CandidateOrigin(
                            operation=operation, input_location=index
                        ),
                    ),
                )
                for index, source in enumerate(sources)
            )
        if not candidates:
            raise StepFailedError(
                PreprocessingFailureCode.NO_CANDIDATES_EXTRACTED,
                "no representation yielded a code candidate",
                evidence=counts,
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(candidates)),
            facts={**counts, "candidate_count": len(candidates)},
        )


__all__ = [
    "CODE_FIELD_NAME",
    "FIELD_MARKER_RE",
    "ExtractAllRepresentations",
    "Representation",
    "field_marker_value",
]
