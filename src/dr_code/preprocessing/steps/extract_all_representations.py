"""Extract candidates additively from every supported representation.

Every representation an LLM response may carry code in is read, and each
one contributes its candidates to a single ordered set — there is no
first-success ladder and no representation shadows another. A response that
is simultaneously a JSON string and a fenced block yields candidates from
both readings; downstream filtering, not extraction order, decides what
survives.

Representations, in the order they contribute candidates:

Segments are read additively too: every fenced block *and* every non-blank
unfenced block contributes, each split at Python anchor lines. Unfenced
code alongside a fenced snippet is a candidate in its own right, so prose
that introduces one solution and fences another yields both.

The two readings that name a code field explicitly come first, so a
response that declares which part is its answer is not shadowed by a
general scrape of some other field:

1. ``json_code_field`` — the ``code`` value of a JSON object, read from
   the whole response and from the fenced blocks of its answer region —
   the ``[[ ## code ## ]]`` field when the response marks one, the whole
   response otherwise — then that value's segments.
2. ``field_marker`` — the value of a ``[[ ## code ## ]]`` field marker,
   then that value's segments.

The remaining readings scrape code out of arbitrary text:

3. ``raw_response`` — the whole normalized text as one candidate.
4. ``text_segments`` — every fenced and every non-blank unfenced block,
   each split at Python anchor lines.
5. ``markdown_segments`` — the same segments with one markdown wrapper
   marker (blockquote, list bullet) stripped per line.
6. ``json_string_response`` — a whole-response JSON string, decoded, then
   re-read as segments.
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

    #: Readings that name a code field explicitly, ahead of the readings
    #: that scrape code out of arbitrary text — see ``_READINGS``.
    JSON_CODE_FIELD = "json_code_field"
    FIELD_MARKER = "field_marker"
    RAW_RESPONSE = "raw_response"
    TEXT_SEGMENTS = "text_segments"
    MARKDOWN_SEGMENTS = "markdown_segments"
    JSON_STRING_RESPONSE = "json_string_response"
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


def _additive_blocks(text: str) -> list[str]:
    """Every fenced block followed by every non-blank unfenced block.

    ``text_analysis.candidate_blocks`` reads fenced blocks *or else* the
    first unfenced one, so unfenced code sitting alongside any fenced
    snippet never becomes a candidate. This step reads both families
    additively instead, which is what an exhaustive definition owes: a
    response carrying code in two places contributes both, and downstream
    filtering decides what survives. ``candidate_blocks`` itself is shared
    with version-pinned metric operators and keeps its own semantics.
    """
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
    return _segment_sources(_additive_blocks(decoded))


def _declared_code_sources(value: str) -> list[str]:
    """A declared code field's value, then its segments.

    A field that names itself ``code`` is a declaration, but its value is
    not always bare source: a response may still wrap the answer in prose
    or fences inside the field. Contributing the value as written *and*
    its segments keeps the declaration usable in both shapes — otherwise
    a wrapped value is unparseable, survives no filter, and a general
    scrape of some earlier field wins the ordinal instead.

    The wrappers a value may carry are the same ones the general readings
    handle — fences, and markdown markers such as a blockquote or a list
    bullet — so both are unwrapped here rather than only fences.

    Order matters: the value as written comes first, so a field holding
    bare source is preferred to any segment split out of it.
    """

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
    """The ``code`` value of ``text`` read as a whole JSON object."""
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, dict):
        return None
    value = decoded.get(CODE_FIELD_NAME)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _envelope_fences(text: str) -> list[str]:
    """The fenced blocks a JSON envelope may be read from.

    The answer region, when the response marks one: a response using field
    markers is answering under ``[[ ## code ## ]]``, and its other fields
    carry material it did not write. Absent that marker — including a
    response with markers but no code field, which declares no answer
    region anywhere — nothing narrows the response and every fence is read.
    """
    region = field_marker_value(text, field_name=CODE_FIELD_NAME)
    return split_by_fences(text if region is None else region)[1]


def _json_code_field(text: str) -> list[str]:
    """A JSON object's ``code`` value, and its segments.

    The object is read from the response as written *and* from each of its
    fenced blocks, because a response that answers in JSON commonly puts
    the envelope inside a fence — ```` ```json ```` or an untagged one —
    and a fenced envelope is the same declaration as a bare one. Reading
    only the bare shape leaves the declaration unread and the response is
    scraped instead, so whatever a general segment reading happens to find
    takes the ordinal the declared field should have held.

    Decoding stays strict: an envelope is read only when the block is a
    complete JSON object carrying a non-blank string ``code``. Nothing is
    repaired, so a truncated or malformed envelope contributes nothing here
    and the scraping readings still get their chance at the response.

    Which fences are scanned is bounded by what the response declares. A
    response carrying field markers has already named its answer region, so
    only that region's fences are read: a ``[[ ## prompt ## ]]`` holding a
    worked example or a reference implementation is context the response was
    given, not an answer it wrote, and reading its envelope would let the
    example take the ordinal ahead of the marked answer — the shadowing this
    reading's position exists to prevent. A response with no markers has
    named no region, so its whole text is the answer and every fence is
    read.
    """
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


#: Representation -> the reading that produces its candidate sources. The
#: mapping's order is the order candidates are contributed in.
#:
#: The two readings that name a code field explicitly — a JSON ``code`` key,
#: fenced or bare, and a ``[[ ## code ## ]]`` marker — come first, ahead of
#: the readings that scrape code out of arbitrary text. A response that says
#: which part is its answer is answering the question directly, while a
#: segment scrape is inference; when both fire, the response's own
#: declaration is the better candidate. Without this, a fenced block in some
#: *other* marked field (a ``[[ ## prompt ## ]]`` carrying a starter or
#: reference function) is scraped first and shadows the marked answer under
#: an acceptance policy that takes the lowest surviving ordinal. Ordering
#: alone does not settle that case: the envelope reading is itself bounded
#: to the answer region, so a non-code field's envelope is not read as the
#: response's declaration either — see ``_envelope_fences``.
#:
#: This orders the readings; it does not make any of them exclusive.
#: Every representation still contributes every candidate it finds.
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
