"""Exhaustively extract code candidates and their discovery provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import ClassVar, Final

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.text_analysis import (
    candidate_blocks,
    code_like_blocks,
    split_by_fences,
)
from dr_code.text_transforms import (
    recover_escaped_python,
    strip_markdown_wrappers,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateLineage,
    CandidateOrigin,
    CodeCandidateSetArtifact,
    TextArtifact,
)


class ExtractionVariant(StrEnum):
    """Decoder-text representations considered in a fixed order."""

    NORMALIZED_RAW_RESPONSE = "normalized_raw_response"
    DECODED_WHOLE_RESPONSE_JSON_STRING = "decoded_whole_response_json_string"
    TOP_LEVEL_JSON_CODE = "top_level_json_code"
    FIELD_MARKER_CODE = "field_marker_code"


class ExtractionStrategy(StrEnum):
    """Discovery rules applied to every applicable text variant."""

    FENCED_BLOCKS = "fenced_blocks"
    MARKDOWN_WRAPPER = "markdown_wrapper"
    ESCAPED_PYTHON = "escaped_python"
    ESCAPED_MARKDOWN_WRAPPER = "escaped_markdown_wrapper"


#: A rule returns every code-like candidate it discovers, in source order.
ExtractionStrategyFn = Callable[[str], list[str]]

_FIELD_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\[\s*##\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*##\s*\]\]"
)


def _code_like_candidates(blocks: list[str]) -> list[str]:
    """Refine blocks exactly as the legacy extraction path did."""
    return [block for block in code_like_blocks(blocks) if block.strip()]


def _fenced_blocks(text: str) -> list[str]:
    """Use every fenced block, or the first unfenced block when none exist."""
    unfenced, fenced = split_by_fences(text)
    if fenced:
        return [block for block in fenced if block.strip()]
    return _code_like_candidates(unfenced[:1])


def _markdown_wrapper(text: str) -> list[str]:
    """Remove one markdown wrapper marker per line before refinement."""
    return _code_like_candidates(
        [strip_markdown_wrappers(block) for block in candidate_blocks(text)]
    )


def _escaped_python(text: str) -> list[str]:
    """Recover structural escapes, then discover candidates from that text."""
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return []
    return _code_like_candidates(candidate_blocks(unescaped))


def _escaped_markdown_wrapper(text: str) -> list[str]:
    """Recover escapes and then remove markdown wrappers before refinement."""
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return []
    return _code_like_candidates(
        [
            strip_markdown_wrappers(block)
            for block in candidate_blocks(unescaped)
        ]
    )


STRATEGY_REGISTRY: dict[str, ExtractionStrategyFn] = {
    ExtractionStrategy.FENCED_BLOCKS.value: _fenced_blocks,
    ExtractionStrategy.MARKDOWN_WRAPPER.value: _markdown_wrapper,
    ExtractionStrategy.ESCAPED_PYTHON.value: _escaped_python,
    ExtractionStrategy.ESCAPED_MARKDOWN_WRAPPER.value: _escaped_markdown_wrapper,
}

DEFAULT_STRATEGIES: Final = (
    ExtractionStrategy.FENCED_BLOCKS,
    ExtractionStrategy.MARKDOWN_WRAPPER,
    ExtractionStrategy.ESCAPED_PYTHON,
    ExtractionStrategy.ESCAPED_MARKDOWN_WRAPPER,
)


class ExtractCandidatesSettings(StepSettings):
    """Ordered discovery rules; each rule runs against every text variant."""

    alternatives: tuple[ExtractionStrategy, ...] = DEFAULT_STRATEGIES


def _json_value(text: str) -> object | None:
    """Parse a whole response as JSON without treating parse failure as data."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None


def _field_marker_code(text: str) -> str | None:
    """Return the code field through the next field marker, if present.

    The next marker is a structural closing delimiter regardless of its field
    name, matching the established ``[[ ## code ## ]]`` response format.
    """
    matches = tuple(_FIELD_MARKER_RE.finditer(text))
    for index, marker in enumerate(matches):
        if marker.group("field") != "code":
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        return text[marker.end() : end]
    return None


def _variants(text: str) -> tuple[tuple[ExtractionVariant, str], ...]:
    """Build every applicable decoder-text representation in stable order."""
    variants: list[tuple[ExtractionVariant, str]] = [
        (ExtractionVariant.NORMALIZED_RAW_RESPONSE, text)
    ]
    decoded = _json_value(text)
    if isinstance(decoded, str):
        variants.append(
            (ExtractionVariant.DECODED_WHOLE_RESPONSE_JSON_STRING, decoded)
        )
    elif isinstance(decoded, Mapping):
        code = decoded.get("code")
        if isinstance(code, str):
            variants.append((ExtractionVariant.TOP_LEVEL_JSON_CODE, code))

    field_value = _field_marker_code(text)
    if field_value is not None:
        variants.append(
            (
                ExtractionVariant.FIELD_MARKER_CODE,
                field_value.strip("\r\n"),
            )
        )
    return tuple(variants)


class ExtractCandidates(Step[ExtractCandidatesSettings]):
    """Emit all candidates from all variants and discovery rules.

    This step records provenance but intentionally does not deduplicate: later
    cleaning can make initially distinct candidates identical, and the dedupe
    step is responsible for merging those origins after that transformation.
    """

    NAME: ClassVar[StepName] = StepName.EXTRACT_CANDIDATES
    VERSION: ClassVar[str] = "2"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    Settings = ExtractCandidatesSettings

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        candidates: list[str] = []
        lineage: list[CandidateLineage] = []
        origin_counts: list[dict[str, str | int]] = []
        variant_counts: list[dict[str, str | int]] = []

        for variant, text in _variants(value.text):
            variant_count = 0
            for strategy in self.settings.alternatives:
                discovered = STRATEGY_REGISTRY[strategy.value](text)
                origin = CandidateOrigin(
                    variant=variant.value, strategy=strategy.value
                )
                candidates.extend(discovered)
                lineage.extend(
                    CandidateLineage(origins=(origin,)) for _ in discovered
                )
                variant_count += len(discovered)
                origin_counts.append(
                    {
                        "variant": variant.value,
                        "strategy": strategy.value,
                        "candidate_count": len(discovered),
                    }
                )
            variant_counts.append(
                {"variant": variant.value, "candidate_count": variant_count}
            )

        facts = {
            "candidate_count": len(candidates),
            "variant_count": len(variant_counts),
            "variants": variant_counts,
            "origins": origin_counts,
        }
        if candidates:
            # Retained for earlier trace consumers; it means first discovery,
            # not a selected alternative.
            facts["alternative"] = lineage[0].origins[0].strategy
            return StepOutput(
                value=CodeCandidateSetArtifact(
                    candidates=tuple(candidates), lineage=tuple(lineage)
                ),
                facts=facts,
            )
        raise StepFailedError(
            "no code candidates extracted",
            failure_code=PreprocessingFailureCode.NO_CODE_CANDIDATES,
            facts=facts,
        )


__all__ = [
    "DEFAULT_STRATEGIES",
    "STRATEGY_REGISTRY",
    "ExtractCandidates",
    "ExtractCandidatesSettings",
    "ExtractionStrategy",
    "ExtractionStrategyFn",
    "ExtractionVariant",
]
