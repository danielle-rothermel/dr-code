"""Transport-neutral values for the preprocessing viewer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
)

_SHA256_LENGTH: Final = 64


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


@dataclass(frozen=True, slots=True)
class Annotation:
    corpus_sha256: str
    sample_id: str
    decoder_output_sha256: str
    verdict: Verdict | None
    note: str | None
    tags: tuple[Tag, ...]


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


__all__ = (
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
    "Verdict",
    "ViewerError",
    "Waterfall",
    "WaterfallStage",
    "validate_sha256",
)
