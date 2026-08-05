"""Extract candidates additively from every supported representation.

Every representation an LLM response may carry code in is read, and each
one contributes its candidates to a single ordered set — there is no
first-success ladder and no representation shadows another. A response that
is simultaneously a JSON string and a fenced block yields candidates from
both readings; downstream filtering, not extraction order, decides what
survives.

Representations, in the order they contribute candidates:

1. ``raw_response`` — the whole normalized text as one candidate.
2. ``text_segments`` — fenced blocks when present, plus the first unfenced
   block, each split at Python anchor lines.
3. ``markdown_segments`` — the same segments with one markdown wrapper
   marker (blockquote, list bullet) stripped per line.
4. ``json_string_response`` — a whole-response JSON string, decoded, then
   re-read as segments.
5. ``json_code_field`` — a top-level JSON object's ``code`` value.
6. ``field_marker`` — the value of a ``[[ ## code ## ]]`` field marker.
7. ``escaped_python`` — structurally escaped Python, recovered and re-read
   as segments.
8. ``escaped_markdown`` — the recovered text with markdown wrappers
   stripped per segment.

Failing to read a representation contributes nothing; it is never an error.
Only a response from which no representation yields any candidate fails.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum, verify, UNIQUE
from typing import ClassVar, Final

from dr_code.text_analysis import candidate_blocks, code_like_blocks
from dr_code.text_transforms import (
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

#: The field-marker syntax ``[[ ## name ## ]]`` delimiting a named region.
FIELD_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)
#: The JSON object key and field-marker name a code value is carried under.
CODE_FIELD_NAME: Final[str] = "code"


@verify(UNIQUE)
class Representation(StrEnum):
    """The representations read, in contribution order.

    Each member's value is the ``operation_name`` stamped on the origins of
    the candidates it produces, so a candidate's lineage names the reading
    that found it.
    """

    RAW_RESPONSE = "raw_response"
    TEXT_SEGMENTS = "text_segments"
    MARKDOWN_SEGMENTS = "markdown_segments"
    JSON_STRING_RESPONSE = "json_string_response"
    JSON_CODE_FIELD = "json_code_field"
    FIELD_MARKER = "field_marker"
    ESCAPED_PYTHON = "escaped_python"
    ESCAPED_MARKDOWN = "escaped_markdown"


def field_marker_value(text: str, *, field_name: str) -> str | None:
    """The text between ``field_name``'s marker and the next marker."""
    matches = list(FIELD_MARKER_RE.finditer(text))
    for index, match in enumerate(matches):
        if match.group("field") != field_name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return text[start:end]
    return None


def _segment_sources(blocks: list[str]) -> list[str]:
    """Split blocks at Python anchors and drop whitespace-only results."""
    return [block for block in code_like_blocks(blocks) if block.strip()]


def _raw_response(text: str) -> list[str]:
    return [text] if text.strip() else []


def _text_segments(text: str) -> list[str]:
    return _segment_sources(candidate_blocks(text))


def _markdown_segments(text: str) -> list[str]:
    return _segment_sources(
        [strip_markdown_wrappers(block) for block in candidate_blocks(text)]
    )


def _json_string_response(text: str) -> list[str]:
    """Segments of a whole-response JSON string, once decoded."""
    stripped = text.strip()
    if not (stripped.startswith('"') and stripped.endswith('"')):
        return []
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, str) or decoded == text:
        return []
    return _segment_sources(candidate_blocks(decoded))


def _json_code_field(text: str) -> list[str]:
    """A top-level JSON object's ``code`` value, when it is a string."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return []
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, dict):
        return []
    value = decoded.get(CODE_FIELD_NAME)
    if not isinstance(value, str) or not value.strip():
        return []
    return [value]


def _field_marker(text: str) -> list[str]:
    value = field_marker_value(text, field_name=CODE_FIELD_NAME)
    if value is None or not value.strip():
        return []
    return [value.strip()]


def _escaped_python(text: str) -> list[str]:
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return []
    return _segment_sources(candidate_blocks(unescaped))


def _escaped_markdown(text: str) -> list[str]:
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return []
    return _segment_sources(
        [
            strip_markdown_wrappers(block)
            for block in candidate_blocks(unescaped)
        ]
    )


#: Representation -> the reading that produces its candidate sources. The
#: mapping's order is the order candidates are contributed in.
_READINGS: Final = (
    (Representation.RAW_RESPONSE, _raw_response),
    (Representation.TEXT_SEGMENTS, _text_segments),
    (Representation.MARKDOWN_SEGMENTS, _markdown_segments),
    (Representation.JSON_STRING_RESPONSE, _json_string_response),
    (Representation.JSON_CODE_FIELD, _json_code_field),
    (Representation.FIELD_MARKER, _field_marker),
    (Representation.ESCAPED_PYTHON, _escaped_python),
    (Representation.ESCAPED_MARKDOWN, _escaped_markdown),
)


class ExtractAllRepresentations(Step[StepSettings]):
    """Text -> CandidateSet, additively across every representation.

    Reads all representations and concatenates their candidates in the
    order declared by ``Representation``. Each candidate's single origin
    names the representation that produced it and the ordinal of the source
    within that representation's own output. Failing to read a
    representation contributes nothing. When no representation yields a
    candidate the step fails with ``NO_CANDIDATES_EXTRACTED``, attaching
    the per-representation counts as evidence.
    """

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
