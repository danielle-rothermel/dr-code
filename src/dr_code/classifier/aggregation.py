"""Descriptive aggregation over repeated classifications."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, Mapping

from dr_code.classifier.taxonomy import OTHER_LABEL

AGGREGATION_VERSION: Final = "failure-aggregation-v1"


class RepeatFailureKind(StrEnum):
    TRANSPORT = "transport"
    INVALID_RESPONSE = "invalid_response"


class RepeatPhase(StrEnum):
    PRIMARY = "primary"
    CORRECTION = "correction"


@dataclass(frozen=True, slots=True)
class RepeatFailure:
    kind: RepeatFailureKind
    detail: str


@dataclass(frozen=True, slots=True)
class RepeatOutcome:
    label: str | None
    rationale: str | None
    failure: RepeatFailure | None = None
    phase: RepeatPhase = RepeatPhase.PRIMARY
    attempt: Literal[1, 2] = 1
    corrected: bool = False
    primary_validation_failure: str | None = None

    def __post_init__(self) -> None:
        succeeded = self.label is not None and self.rationale is not None
        if succeeded == (self.failure is not None):
            raise ValueError(
                "repeat outcome must contain either a response or a failure"
            )
        if self.phase is RepeatPhase.PRIMARY:
            if (
                self.attempt != 1
                or self.corrected
                or self.primary_validation_failure is not None
            ):
                raise ValueError(
                    "primary repeat outcome has invalid correction metadata"
                )
        elif (
            self.attempt != 2
            or self.primary_validation_failure is None
            or self.corrected != succeeded
        ):
            raise ValueError(
                "correction repeat outcome has invalid correction metadata"
            )

    @property
    def succeeded(self) -> bool:
        return self.failure is None


@dataclass(frozen=True, slots=True)
class ItemAggregate:
    label: str | None
    agreement: float | None
    tie: bool
    successful_repeats: int
    failed_repeats: int
    label_counts: Mapping[str, int]


def aggregate_repeats(
    outcomes: tuple[RepeatOutcome, ...] | list[RepeatOutcome],
) -> ItemAggregate:
    labels = [outcome.label for outcome in outcomes if outcome.succeeded]
    if not labels:
        return ItemAggregate(
            label=None,
            agreement=None,
            tie=False,
            successful_repeats=0,
            failed_repeats=len(outcomes),
            label_counts=MappingProxyType({}),
        )
    counts = Counter(label for label in labels if label is not None)
    top_count = max(counts.values())
    winners = tuple(
        sorted(label for label, count in counts.items() if count == top_count)
    )
    tie = len(winners) > 1
    label = OTHER_LABEL if tie else winners[0]
    return ItemAggregate(
        label=label,
        agreement=top_count / len(labels),
        tie=tie,
        successful_repeats=len(labels),
        failed_repeats=len(outcomes) - len(labels),
        label_counts=MappingProxyType(dict(sorted(counts.items()))),
    )


def mean_agreement(
    aggregates: tuple[ItemAggregate, ...] | list[ItemAggregate],
) -> float | None:
    values = tuple(
        item.agreement for item in aggregates if item.agreement is not None
    )
    return sum(values) / len(values) if values else None


__all__ = (
    "AGGREGATION_VERSION",
    "ItemAggregate",
    "RepeatFailure",
    "RepeatFailureKind",
    "RepeatOutcome",
    "RepeatPhase",
    "aggregate_repeats",
    "mean_agreement",
)
