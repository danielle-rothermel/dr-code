"""Extraction — Raw LLM Output to Code Candidates.

This module owns Text Normalization, extractor ordering, raw/text-normalized
fan-out, and extractor_path Attribution.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Callable
from typing import Final

from dr_code.text_transforms import (
    is_code_anchor_line,
    is_code_like_line,
    normalize_text,
    strip_markdown_wrappers,
)

from code_eval.config import ValidatorConfig
from code_eval.models.extracted_candidate import ExtractedCandidate
from code_eval.models.extraction_fragment import ExtractionFragment
from code_eval.models.extraction_pass import ExtractionPass
from code_eval.models.extraction_result import ExtractionResult
from code_eval.models.extraction_step import ExtractionStep
from code_eval.names import DEFAULT_TAB_WIDTH, ExtractorName

_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<open>```|~~~)(?P<tag>[A-Za-z0-9_+\-]*)\s*\n(?P<body>.*?)(?P<close>(?P=open))",
    flags=re.DOTALL,
)
_OPEN_FENCE_AT_END: Final[re.Pattern[str]] = re.compile(
    r"(?P<open>```|~~~)(?P<tag>[A-Za-z0-9_+\-]*)\s*\n(?P<body>(?:(?!```|~~~).)*)\Z",
    flags=re.DOTALL,
)
_LEAD_IN: Final[re.Pattern[str]] = re.compile(
    r"(?i)^.*\b(?:here(?:'s| is)|here)\s+(?:is\s+)?(?:the|my|a|an)?\s*"
    r"(?:solution|code|answer|program|implementation)\b[:.]?\s*$"
)
_CODE_LIKE_PROSE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:from |import |def |class |@|async def |if |for |while |with |try |return )"
)
_INLINE_BACKTICK_RE: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")

type ExtractorFunction = Callable[[str], tuple[ExtractionFragment, ...]]


def text_normalize(raw: str, tab_width: int = DEFAULT_TAB_WIDTH) -> str:
    return normalize_text(raw, tab_width)


def _direct_parse(raw: str) -> tuple[ExtractionFragment, ...]:
    out = [ExtractionFragment(source=raw)]
    dedented = textwrap.dedent(raw)
    if dedented != raw:
        out.append(
            ExtractionFragment(
                source=dedented,
                notes="textwrap.dedent applied",
                emitted_as=ExtractorName.DIRECT_PARSE_DEDENTED,
            )
        )
    return tuple(out)


def _fences(raw: str) -> tuple[ExtractionFragment, ...]:
    out: list[ExtractionFragment] = []
    for match in _FENCE_RE.finditer(raw):
        out.append(
            ExtractionFragment(
                source=match.group("body"),
                notes=f"tag={match.group('tag') or 'none'!r}",
            )
        )

    if out:
        return tuple(out)

    tail = _OPEN_FENCE_AT_END.search(raw)
    if tail is None:
        return ()
    return (
        ExtractionFragment(
            source=tail.group("body"),
            notes=f"unterminated; tag={tail.group('tag') or 'none'!r}",
        ),
    )


def _keyword_anchor(raw: str) -> tuple[ExtractionFragment, ...]:
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if is_code_anchor_line(line):
            return (
                ExtractionFragment(
                    source="\n".join(lines[index:]),
                    notes=f"anchored at line {index}",
                ),
            )
    return ()


def _trim_trailing_prose(text: str) -> str:
    lines = text.splitlines()
    while lines:
        candidate = "\n".join(lines)
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            lines.pop()
    return text


def _prose_patterns(raw: str) -> tuple[ExtractionFragment, ...]:
    lines = raw.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if _LEAD_IN.match(line.strip()):
            start = index + 1
            break

    if start is None:
        for index, line in enumerate(lines):
            if _CODE_LIKE_PROSE.match(line):
                start = index
                break

    if start is None:
        return ()

    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines):
        return ()

    body = "\n".join(lines[start:])
    note = f"lead-in at line {start - 1}" if start > 0 else ""
    return (ExtractionFragment(source=_trim_trailing_prose(body), notes=note),)


def _is_code_like(line: str) -> bool:
    return is_code_like_line(line)


def _indentation_block(raw: str) -> tuple[ExtractionFragment, ...]:
    lines = raw.splitlines()
    best: tuple[int, int] | None = None
    run_start: int | None = None

    for index, line in enumerate(lines):
        if _is_code_like(line):
            if run_start is None:
                run_start = index
            if best is None or index - run_start + 1 > best[1] - best[0]:
                best = (run_start, index + 1)
        else:
            run_start = None

    if best is None:
        return ()

    start, end = best
    while end > start and not lines[end - 1].strip():
        end -= 1
    while start < end and not lines[start].strip():
        start += 1
    if start >= end:
        return ()

    return (
        ExtractionFragment(
            source="\n".join(lines[start:end]), notes=f"lines [{start}, {end})"
        ),
    )


def _markdown_strip(raw: str) -> tuple[ExtractionFragment, ...]:
    stripped = strip_markdown_wrappers(raw)
    if stripped == "\n".join(raw.splitlines()):
        return ()
    return (ExtractionFragment(source=stripped),)


def _inline_spans(raw: str) -> tuple[ExtractionFragment, ...]:
    stripped = raw.strip()
    if (
        stripped.startswith("`")
        and stripped.endswith("`")
        and not stripped.startswith("```")
    ):
        return (
            ExtractionFragment(
                source=stripped[1:-1], notes="full-source backtick wrap"
            ),
        )
    if "`" not in raw or "```" in raw:
        return ()
    stripped_inline = _INLINE_BACKTICK_RE.sub(r"\1", raw)
    if stripped_inline == raw:
        return ()
    return (
        ExtractionFragment(
            source=stripped_inline, notes="inline backticks removed"
        ),
    )


EXTRACTION_CATALOG: Final[
    tuple[tuple[ExtractorName, ExtractorFunction], ...]
] = (
    (ExtractorName.DIRECT_PARSE, _direct_parse),
    (ExtractorName.FENCES, _fences),
    (ExtractorName.KEYWORD_ANCHOR, _keyword_anchor),
    (ExtractorName.PROSE_PATTERNS, _prose_patterns),
    (ExtractorName.INDENTATION_BLOCK, _indentation_block),
    (ExtractorName.MARKDOWN_STRIP, _markdown_strip),
    (ExtractorName.INLINE_SPANS, _inline_spans),
)


def _attach(
    extractor: ExtractorName,
    fragments: tuple[ExtractionFragment, ...],
    *,
    text_normalized: bool,
) -> tuple[ExtractedCandidate, ...]:
    out: list[ExtractedCandidate] = []
    for fragment in fragments:
        emitted_as = fragment.emitted_as or extractor
        path = (emitted_as.value,)
        if text_normalized:
            path = (ExtractorName.TEXT_NORMALIZE.value, *path)
        out.append(
            ExtractedCandidate(
                source=fragment.source,
                extractor=emitted_as,
                extractor_path=path,
                notes=fragment.notes,
            )
        )
    return tuple(out)


def run_extraction(raw: str, config: ValidatorConfig) -> ExtractionResult:
    """Run Text Normalization and Extraction over Raw LLM Output."""
    normalized = text_normalize(raw, tab_width=config.tab_width)
    extraction_log: list[ExtractionStep] = [
        ExtractionStep(
            extractor=ExtractorName.TEXT_NORMALIZE,
            candidates_produced=1,
            notes="raw -> normalized",
        )
    ]
    candidates: list[ExtractedCandidate] = []
    passes: list[ExtractionPass] = []

    for extractor, extract in EXTRACTION_CATALOG:
        raw_candidates = _attach(
            extractor, extract(raw), text_normalized=False
        )
        normalized_candidates = _attach(
            extractor, extract(normalized), text_normalized=True
        )
        produced = (*raw_candidates, *normalized_candidates)
        candidates.extend(produced)
        passes.append(
            ExtractionPass(
                extractor=extractor,
                raw_candidates=raw_candidates,
                normalized_candidates=normalized_candidates,
            )
        )
        extraction_log.append(
            ExtractionStep(
                extractor=extractor,
                candidates_produced=len(produced),
                notes=f"raw={len(raw_candidates)}, normalized={len(normalized_candidates)}",
            )
        )

    return ExtractionResult(
        normalized_output=normalized,
        candidates=tuple(candidates),
        passes=tuple(passes),
        extraction_log=tuple(extraction_log),
    )


__all__ = ["EXTRACTION_CATALOG", "run_extraction", "text_normalize"]
