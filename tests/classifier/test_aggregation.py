from __future__ import annotations

import pytest

from dr_code.classifier.aggregation import (
    RepeatOutcome,
    aggregate_repeats,
    mean_agreement,
)


def _ok(label: str) -> RepeatOutcome:
    return RepeatOutcome(label=label, rationale="because")


def _fail() -> RepeatOutcome:
    return RepeatOutcome(
        label=None, rationale=None, failure_reason="malformed-response"
    )


def test_clear_majority_wins_with_hand_computed_agreement() -> None:
    outcomes = [
        _ok("truncated-output"),
        _ok("truncated-output"),
        _ok("truncated-output"),
        _ok("prose-no-code"),
        _ok("prose-no-code"),
    ]
    result = aggregate_repeats(outcomes)
    assert result.majority_label == "truncated-output"
    assert result.tie is False
    # 3 of 5 successful repeats agree with the winner.
    assert result.agreement == pytest.approx(3 / 5)
    assert result.successful_repeats == 5
    assert result.failed_repeats == 0
    assert result.label_counts == {
        "prose-no-code": 2,
        "truncated-output": 3,
    }


def test_tie_resolves_to_other_and_records_the_tie() -> None:
    outcomes = [_ok("truncated-output"), _ok("prose-no-code")]
    result = aggregate_repeats(outcomes)
    assert result.majority_label == "other"
    assert result.tie is True
    # 'other' received zero votes, so agreement with the winner is 0.
    assert result.agreement == pytest.approx(0.0)


def test_tie_that_includes_other_reports_its_share() -> None:
    outcomes = [_ok("other"), _ok("prose-no-code")]
    result = aggregate_repeats(outcomes)
    assert result.majority_label == "other"
    assert result.tie is True
    assert result.agreement == pytest.approx(1 / 2)


def test_failed_repeats_do_not_vote_but_lower_nothing() -> None:
    outcomes = [_ok("prose-no-code"), _ok("prose-no-code"), _fail()]
    result = aggregate_repeats(outcomes)
    assert result.majority_label == "prose-no-code"
    # Agreement is over the two successful repeats only.
    assert result.agreement == pytest.approx(1.0)
    assert result.successful_repeats == 2
    assert result.failed_repeats == 1


def test_all_repeats_failed_yields_no_majority() -> None:
    result = aggregate_repeats([_fail(), _fail()])
    assert result.majority_label is None
    assert result.agreement is None
    assert result.successful_repeats == 0
    assert result.failed_repeats == 2


def test_mean_agreement_skips_items_without_a_label() -> None:
    a = aggregate_repeats([_ok("a"), _ok("a")])
    b = aggregate_repeats([_ok("a"), _ok("b")])  # tie -> other, agreement 0
    c = aggregate_repeats([_fail()])  # no label
    assert mean_agreement([a, b, c]) == pytest.approx((1.0 + 0.0) / 2)
    assert mean_agreement([c]) is None
