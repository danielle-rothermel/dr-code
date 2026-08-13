from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.trace import Absent, TextArtifact, is_absent


def test_absent_preserves_causal_lineage() -> None:
    absent = Absent(
        failed_step="parse",
        failure_code="no_candidate_survived_filtering",
        cause="syntax error",
        propagated_through=("score", "aggregate"),
    )

    assert absent.model_dump(mode="json") == {
        "kind": "absent",
        "failed_step": "parse",
        "failure_code": "no_candidate_survived_filtering",
        "cause": "syntax error",
        "propagated_through": ["score", "aggregate"],
    }
    with pytest.raises(ValidationError):
        absent.cause = "other"  # type: ignore[misc]


def test_absent_requires_a_failure_code() -> None:
    with pytest.raises(ValidationError):
        Absent(failed_step="parse", cause="syntax error")  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (
            Absent(
                failed_step="parse",
                failure_code="parse_failed",
                cause="syntax error",
            ),
            True,
        ),
        (TextArtifact(text="present"), False),
        ("not a trace value", False),
        (None, False),
    ),
)
def test_is_absent_distinguishes_causal_absence(
    value: object,
    expected: bool,
) -> None:
    assert is_absent(value) is expected
