"""Transport-neutral values for the preprocessing viewer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
)

_SHA256_LENGTH: Final = 64
ANNOTATION_NOTE_MAX_LENGTH: Final = 10_000
ANNOTATION_TAG_IDS_MAX_COUNT: Final = 100
TAG_NAME_MAX_LENGTH: Final = 100
# Persisted tag identity contract; mirror these literals in the TypeScript UI.
TAG_NAME_WHITESPACE_CODE_POINTS: Final = (
    0x0009,
    0x000A,
    0x000B,
    0x000C,
    0x000D,
    0x0020,
    0x0085,
    0x00A0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
)
_TAG_NAME_WHITESPACE = frozenset(TAG_NAME_WHITESPACE_CODE_POINTS)
_SURROGATE_MIN: Final = 0xD800
_SURROGATE_MAX: Final = 0xDFFF


class ViewerError(ValueError):
    """Base error for invalid viewer state or requests."""


class RunNotFoundError(ViewerError):
    """A requested run ID is not registered."""


class InvalidQueryError(ViewerError):
    """An analytical query received invalid parameters."""


class IncompatibleRunsError(ViewerError):
    """Two runs cannot be compared without misleading results."""


class Verdict(StrEnum):
    SHOULD_BE_PARSEABLE = "should_be_parseable"
    EXPECTED_NO_CODE = "expected_no_code"


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    label: str
    manifest_sha256: str
    corpus_sha256: str
    definition_id: str
    definition_version: str
    has_evaluation: bool
    definition_identity: str


@dataclass(frozen=True, slots=True)
class WaterfallStage:
    stage_id: str
    label: str
    unit: str
    order: int
    count: int
    denominator: int
    rate: float | None


@dataclass(frozen=True, slots=True)
class Waterfall:
    run: RunSummary
    stages: tuple[WaterfallStage, ...]


@dataclass(frozen=True, slots=True)
class FailureGroup:
    failure_code: str
    failed_step: str
    cause: str | None
    count: int


@dataclass(frozen=True, slots=True)
class Failures:
    run: RunSummary
    groups: tuple[FailureGroup, ...]
    total_count: int


@dataclass(frozen=True, slots=True)
class ExampleSummary:
    sample_id: str
    task_id: str | None
    decoder_output_sha256: str | None
    outcome: str
    failure_code: str | None
    failed_step: str | None
    decoder_output: str | None
    annotation_verdict: Verdict | None


@dataclass(frozen=True, slots=True)
class Page:
    items: tuple[ExampleSummary, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class Tag:
    tag_id: str
    name: str

    def __post_init__(self) -> None:
        _normalized_name, display_name = normalize_tag_name(self.name)
        if self.name != display_name:
            raise InvalidQueryError(
                "tag name must use normalized display whitespace"
            )


@dataclass(frozen=True, slots=True)
class Annotation:
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str
    verdict: Verdict | None
    note: str | None
    tags: tuple[Tag, ...]

    def __post_init__(self) -> None:
        validate_annotation_note(self.note)
        normalize_annotation_tag_ids(tag.tag_id for tag in self.tags)


@dataclass(frozen=True, slots=True)
class ExampleDetail:
    sample_id: str
    corpus_sha256: str
    decoder_output_sha256: str | None
    context: dict[str, object]
    outcome: str
    failure_code: str | None
    failed_step: str | None
    cause: str | None
    raw_decoder_output: str | None
    candidates: tuple[dict[str, object], ...]
    facts: tuple[dict[str, object], ...]
    rejections: tuple[dict[str, object], ...]
    annotation: Annotation | None


@dataclass(frozen=True, slots=True)
class ReviewPage:
    items: tuple[ExampleDetail, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class ComparisonStage:
    stage_id: str
    label: str
    unit: str
    baseline_count: int
    baseline_denominator_count: int
    candidate_count: int
    candidate_denominator_count: int
    count_delta: int
    baseline_rate: float | None
    candidate_rate: float | None
    rate_delta: float | None


@dataclass(frozen=True, slots=True)
class OutcomeTransition:
    baseline_outcome: str
    candidate_outcome: str
    count: int


@dataclass(frozen=True, slots=True)
class RunComparison:
    baseline: RunSummary
    candidate: RunSummary
    stages: tuple[ComparisonStage, ...]
    transitions: tuple[OutcomeTransition, ...]


def validate_sha256(value: str, label: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise InvalidQueryError(f"{label} must be a lowercase SHA-256 digest")
    return value


def normalize_tag_name(name: str) -> tuple[str, str]:
    """Return the persisted comparison and display forms of a tag name."""
    if not isinstance(name, str):
        raise InvalidQueryError("tag name must be a string")
    _validate_unicode_scalar_text(name, "tag name")
    display_name = " ".join(
        part
        for part in "".join(
            " " if ord(character) in _TAG_NAME_WHITESPACE else character
            for character in name
        ).split(" ")
        if part
    )
    if not display_name:
        raise InvalidQueryError("tag name must not be blank")
    if len(display_name) > TAG_NAME_MAX_LENGTH:
        raise InvalidQueryError(
            "tag name must be at most "
            f"{TAG_NAME_MAX_LENGTH} characters after normalization"
        )
    return display_name.casefold(), display_name


def validate_annotation_note(note: str | None) -> str | None:
    if note is None:
        return None
    if not isinstance(note, str):
        raise InvalidQueryError("annotation note must be a string or null")
    _validate_unicode_scalar_text(note, "annotation note")
    if len(note) > ANNOTATION_NOTE_MAX_LENGTH:
        raise InvalidQueryError(
            "annotation note must be at most "
            f"{ANNOTATION_NOTE_MAX_LENGTH} characters"
        )
    return note


def normalize_annotation_tag_ids(tag_ids: Iterable[str]) -> tuple[str, ...]:
    values = tuple(tag_ids)
    if any(not isinstance(tag_id, str) or not tag_id for tag_id in values):
        raise InvalidQueryError("tag_ids must contain nonblank strings")
    for tag_id in values:
        _validate_unicode_scalar_text(tag_id, "tag_ids")
    distinct = tuple(sorted(set(values)))
    if len(distinct) > ANNOTATION_TAG_IDS_MAX_COUNT:
        raise InvalidQueryError(
            "annotation must have at most "
            f"{ANNOTATION_TAG_IDS_MAX_COUNT} distinct tag IDs"
        )
    return distinct


def _validate_unicode_scalar_text(value: str, label: str) -> None:
    if any(
        _SURROGATE_MIN <= ord(character) <= _SURROGATE_MAX
        for character in value
    ):
        raise InvalidQueryError(
            f"{label} must contain only Unicode scalar values"
        )


__all__ = (
    "ANNOTATION_NOTE_MAX_LENGTH",
    "ANNOTATION_TAG_IDS_MAX_COUNT",
    "Annotation",
    "ExampleDetail",
    "ExampleSummary",
    "FailureGroup",
    "Failures",
    "IncompatibleRunsError",
    "InvalidQueryError",
    "OutcomeTransition",
    "Page",
    "ReviewPage",
    "RunComparison",
    "RunDescriptor",
    "RunNotFoundError",
    "RunSummary",
    "RunValidationError",
    "Tag",
    "TAG_NAME_MAX_LENGTH",
    "TAG_NAME_WHITESPACE_CODE_POINTS",
    "Verdict",
    "ViewerError",
    "Waterfall",
    "WaterfallStage",
    "normalize_annotation_tag_ids",
    "normalize_tag_name",
    "validate_annotation_note",
    "validate_sha256",
)
