"""Pure, modular candidate extraction with ordered provenance paths."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from dr_code.fenced_text import TextSegment, extract_fenced_document
from dr_code.text_analysis import anchored_code_blocks
from dr_code.text_transforms import (
    MARKDOWN_WRAPPER_RE,
    recover_escaped_python,
    strip_markdown_wrappers,
)
from dr_code.trace import CandidateOrigin, ExtractionOperation


_FIELD_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)
_MAX_INTERPRETATION_CHARS: Final = 1_000_000


def _ignore_json_number(_value: str) -> None:
    """Avoid converting JSON numbers that extraction never inspects."""
    return None


@dataclass(frozen=True, slots=True)
class TextFragment:
    """One response interpretation and the path that produced it."""

    text: str
    path: tuple[ExtractionOperation, ...]


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    """An extracted source candidate aligned with complete provenance."""

    source: str
    origin: CandidateOrigin


def _operation(kind: str, **details: object) -> ExtractionOperation:
    return ExtractionOperation.model_validate(
        {"kind": kind, "details": details}
    )


def _response_fragment(name: str, text: str) -> TextFragment:
    return TextFragment(
        text=text,
        path=(_operation("response_representation", name=name),),
    )


def _json_value(text: str) -> object | None:
    if len(text) > _MAX_INTERPRETATION_CHARS:
        return None
    try:
        return json.loads(
            text,
            parse_int=_ignore_json_number,
            parse_float=_ignore_json_number,
        )
    except (ValueError, RecursionError):
        return None


def _completed_json_code(text: str) -> str | None:
    """Recover only an EOF-truncated top-level JSON ``code`` string.

    The repair supplies the representation envelope's final quote and brace;
    it never supplies Python source.  Strict JSON decoding and shape checks
    remain authoritative.
    """
    if not text.lstrip().startswith("{") or len(text) < 2:
        return None
    if _json_value(text) is not None:
        return None
    decoded = _json_value(f'{text}"}}')
    if not isinstance(decoded, Mapping) or set(decoded) != {"code"}:
        return None
    code = decoded.get("code")
    return code if isinstance(code, str) else None


def _field_marker_code(text: str) -> str | None:
    matches = tuple(_FIELD_MARKER_RE.finditer(text))
    for index, marker in enumerate(matches):
        if marker.group("field") != "code":
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return text[marker.end() : end].strip("\r\n")
    return None


def response_fragments(text: str) -> tuple[TextFragment, ...]:
    """Return additive response representations in conservative order."""
    fragments = [_response_fragment("normalized_raw_response", text)]
    decoded = _json_value(text)
    if isinstance(decoded, str):
        fragments.append(
            _response_fragment("decoded_whole_response_json_string", decoded)
        )
    elif isinstance(decoded, Mapping):
        code = decoded.get("code")
        if isinstance(code, str):
            fragments.append(_response_fragment("top_level_json_code", code))
    else:
        completed_code = _completed_json_code(text)
        if completed_code is not None:
            fragments.append(
                _response_fragment(
                    "completed_top_level_json_code", completed_code
                )
            )

    field_code = _field_marker_code(text)
    if field_code is not None:
        fragments.append(_response_fragment("field_marker_code", field_code))
    return tuple(fragments)


def _recovered_fragments(fragment: TextFragment) -> tuple[TextFragment, ...]:
    fragments = [fragment]
    recovered = recover_escaped_python(fragment.text)
    if recovered is not None and recovered != fragment.text:
        fragments.append(
            TextFragment(
                text=recovered,
                path=(*fragment.path, _operation("escaped_python_recovery")),
            )
        )
    return tuple(fragments)


def _segment_path(
    path: tuple[ExtractionOperation, ...], segment: TextSegment
) -> tuple[ExtractionOperation, ...]:
    if segment.fence is None:
        return (
            *path,
            _operation(
                "unfenced_segment",
                index=segment.index,
            ),
        )
    fence = segment.fence
    return (
        *path,
        _operation(
            "fenced_block",
            index=fence.index,
            segment_index=segment.index,
            marker=fence.marker,
            tag=fence.tag,
            closed=fence.closed,
        ),
    )


def _fenced_json_code(segment: TextSegment) -> str | None:
    fence = segment.fence
    if fence is None:
        return None
    stripped = segment.content.strip()
    if fence.tag != "json" and not (
        stripped.startswith("{") and stripped.endswith("}")
    ):
        return None
    decoded = _json_value(stripped)
    if not isinstance(decoded, Mapping):
        return None
    code = decoded.get("code")
    return code if isinstance(code, str) else None


def _singleton_string_container(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) > _MAX_INTERPRETATION_CHARS or not stripped.startswith(
        ("[", "(")
    ):
        return None
    try:
        decoded = ast.literal_eval(stripped)
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None
    if (
        isinstance(decoded, (list, tuple))
        and len(decoded) == 1
        and isinstance(decoded[0], str)
    ):
        return decoded[0]
    return None


def _python_drafts(
    text: str,
    path: tuple[ExtractionOperation, ...],
    *,
    emit_whole: bool,
) -> list[CandidateDraft]:
    drafts: list[CandidateDraft] = []
    if emit_whole and text.strip():
        drafts.append(
            CandidateDraft(
                source=text,
                origin=CandidateOrigin(
                    path=(*path, _operation("raw_fenced_block"))
                ),
            )
        )

    forms = [(text, path)]
    has_wrapper = any(
        MARKDOWN_WRAPPER_RE.match(line) for line in text.splitlines()
    )
    if has_wrapper:
        unwrapped = strip_markdown_wrappers(text)
        forms.append(
            (unwrapped, (*path, _operation("markdown_wrapper_removal")))
        )

    for form, form_path in forms:
        stripped = form.strip()
        if stripped.startswith("lambda") and (
            len(stripped) == len("lambda")
            or stripped[len("lambda")].isspace()
            or stripped[len("lambda")] == ":"
        ):
            drafts.append(
                CandidateDraft(
                    source=stripped,
                    origin=CandidateOrigin(
                        path=(*form_path, _operation("standalone_lambda"))
                    ),
                )
            )
        for index, source in enumerate(anchored_code_blocks(form)):
            if not source.strip():
                continue
            drafts.append(
                CandidateDraft(
                    source=source,
                    origin=CandidateOrigin(
                        path=(
                            *form_path,
                            _operation("anchored_python_block", index=index),
                        )
                    ),
                )
            )
    return drafts


def _rediscover_interpreted(
    text: str, path: tuple[ExtractionOperation, ...]
) -> list[CandidateDraft]:
    """Re-enter structural discovery once without further interpretation."""
    drafts: list[CandidateDraft] = []
    document = extract_fenced_document(text)
    for segment in document.segments:
        if not segment.content.strip():
            continue
        segment_path = _segment_path(path, segment)
        drafts.extend(
            _python_drafts(
                segment.content,
                segment_path,
                emit_whole=segment.fence is not None,
            )
        )
    return drafts


def _fragment_drafts(fragment: TextFragment) -> list[CandidateDraft]:
    drafts: list[CandidateDraft] = []
    document = extract_fenced_document(fragment.text)
    for segment in document.segments:
        if not segment.content.strip():
            continue
        path = _segment_path(fragment.path, segment)
        drafts.extend(
            _python_drafts(
                segment.content,
                path,
                emit_whole=segment.fence is not None,
            )
        )

        json_code = _fenced_json_code(segment)
        if json_code is not None:
            drafts.extend(
                _rediscover_interpreted(
                    json_code, (*path, _operation("fenced_json_code"))
                )
            )

        singleton = _singleton_string_container(segment.content)
        if singleton is not None:
            drafts.extend(
                _rediscover_interpreted(
                    singleton,
                    (*path, _operation("singleton_string_container")),
                )
            )
    return drafts


def extract_candidate_drafts(text: str) -> tuple[CandidateDraft, ...]:
    """Extract every bounded candidate interpretation in stable order."""
    drafts: list[CandidateDraft] = []
    for response in response_fragments(text):
        for fragment in _recovered_fragments(response):
            drafts.extend(_fragment_drafts(fragment))
    return tuple(drafts)


__all__ = (
    "CandidateDraft",
    "TextFragment",
    "extract_candidate_drafts",
    "response_fragments",
)
