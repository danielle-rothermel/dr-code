"""Aggregate N per-item repeats into a majority label with agreement.

Rules (see the track brief):

* N repeats per item; the majority label wins.
* A tie for the top count resolves to ``other`` and records that a tie
  occurred.
* Per-item agreement is the fraction of *successful* repeats that agree with
  the winning label. It is a descriptive statistic, never a gate.
* Repeats that produced no label (typed lane failures) do not vote and are
  counted separately; an item with zero successful repeats has no majority
  label and agreement ``None``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from dr_code.classifier.taxonomy import OTHER_LABEL


@dataclass(frozen=True, slots=True)
class RepeatOutcome:
    """One repeat's result: a label + rationale, or a typed failure."""

    label: str | None
    rationale: str | None
    failure_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.label is not None


@dataclass(frozen=True, slots=True)
class ItemAggregate:
    """The aggregated verdict for one failure item across its repeats."""

    majority_label: str | None
    agreement: float | None
    tie: bool
    successful_repeats: int
    failed_repeats: int
    label_counts: dict[str, int] = field(default_factory=dict)


def aggregate_repeats(outcomes: list[RepeatOutcome]) -> ItemAggregate:
    """Reduce one item's repeat outcomes to a majority verdict."""
    labels = [o.label for o in outcomes if o.label is not None]
    failed = sum(1 for o in outcomes if not o.succeeded)
    if not labels:
        return ItemAggregate(
            majority_label=None,
            agreement=None,
            tie=False,
            successful_repeats=0,
            failed_repeats=failed,
            label_counts={},
        )
    counts = Counter(labels)
    top_count = max(counts.values())
    winners = sorted(
        label for label, count in counts.items() if count == top_count
    )
    tie = len(winners) > 1
    majority = OTHER_LABEL if tie else winners[0]
    # Agreement is measured against the winning label. On a tie the winner is
    # 'other'; agreement then reflects how many repeats actually voted 'other'.
    agreeing = counts.get(majority, 0)
    agreement = agreeing / len(labels)
    return ItemAggregate(
        majority_label=majority,
        agreement=agreement,
        tie=tie,
        successful_repeats=len(labels),
        failed_repeats=failed,
        label_counts=dict(sorted(counts.items())),
    )


def mean_agreement(aggregates: list[ItemAggregate]) -> float | None:
    """Mean per-item agreement over items that produced a majority label."""
    values = [a.agreement for a in aggregates if a.agreement is not None]
    if not values:
        return None
    return sum(values) / len(values)


__all__ = (
    "ItemAggregate",
    "RepeatOutcome",
    "aggregate_repeats",
    "mean_agreement",
)
