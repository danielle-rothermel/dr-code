from __future__ import annotations

import pytest

from dr_code.classifier.aggregation import (
    RepeatFailure,
    RepeatFailureKind,
    RepeatOutcome,
    aggregate_repeats,
    mean_agreement,
)


def _ok(label: str) -> RepeatOutcome:
    return RepeatOutcome(label=label, rationale="reason")


def _failed() -> RepeatOutcome:
    return RepeatOutcome(
        label=None,
        rationale=None,
        failure=RepeatFailure(
            RepeatFailureKind.TRANSPORT,
            "timeout",
        ),
    )


def test_majority_and_partial_failure_agreement() -> None:
    result = aggregate_repeats([_ok("a"), _ok("a"), _ok("b"), _failed()])
    assert result.label == "a"
    assert result.agreement == pytest.approx(2 / 3)
    assert result.successful_repeats == 3
    assert result.failed_repeats == 1


def test_tie_uses_other_but_agreement_is_top_vote_share() -> None:
    result = aggregate_repeats([_ok("a"), _ok("b"), _failed()])
    assert result.label == "other"
    assert result.tie is True
    assert result.agreement == pytest.approx(1 / 2)


def test_all_failed_has_no_machine_verdict() -> None:
    result = aggregate_repeats([_failed(), _failed()])
    assert result.label is None
    assert result.agreement is None
    assert result.label_counts == {}


def test_mean_agreement_skips_all_failed_items() -> None:
    clear = aggregate_repeats([_ok("a"), _ok("a")])
    tie = aggregate_repeats([_ok("a"), _ok("b")])
    failed = aggregate_repeats([_failed()])
    assert mean_agreement([clear, tie, failed]) == pytest.approx(0.75)
    assert mean_agreement([failed]) is None
