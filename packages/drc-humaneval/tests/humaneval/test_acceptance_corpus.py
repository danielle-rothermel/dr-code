from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest

from drc_humaneval.acceptance import (
    extract_humaneval_code,
    humaneval_runner,
)

_FIXTURE: Final[Path] = Path(__file__).parent / "corpus" / "hard_examples.json"


XFAIL_REASONS: Final[dict[str, str]] = {
    "annotation-1b961da8b9d59f41": (
        "lambda-only solution: bare lambdas are not rendered as named "
        "functions, so no top-level function survives"
    ),
    "annotation-1f427a1043acf297": (
        "lambda-only solution: bare lambdas are not rendered as named "
        "functions, so no top-level function survives"
    ),
    "annotation-72944a2187a55d01": (
        "singleton string container: singleton string containers are not "
        "unwrapped, so the container never compiles"
    ),
    "annotation-b0934e569fa9de7d": (
        "JSON envelope truncated at EOF: unterminated JSON envelopes are not "
        "completed, and strict decoding rejects them"
    ),
    "annotation-d5ed37d4dadbbc75": (
        "JSON envelope truncated at EOF: unterminated JSON envelopes are not "
        "completed, and strict decoding rejects them"
    ),
}


def _load_cases() -> list[dict[str, Any]]:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    cases = document["cases"]
    assert isinstance(cases, list) and cases
    return cases


CASES: Final[list[dict[str, Any]]] = _load_cases()


def _case_id(case: dict[str, Any]) -> str:
    return f"{case['partition']}-{case['id']}"


def _parametrized() -> list[Any]:
    params = []
    for case in CASES:
        reason = XFAIL_REASONS.get(case["id"])
        marks = (
            [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        )
        params.append(pytest.param(case, id=_case_id(case), marks=marks))
    return params


@pytest.fixture(scope="module")
def runner():
    return humaneval_runner()


@pytest.mark.parametrize("case", _parametrized())
def test_extraction_matches_the_recorded_verdict(
    case: dict[str, Any], runner
) -> None:
    result = extract_humaneval_code(case["decoder_output"], runner=runner)

    if case["expected_outcome"] == "absent":
        assert not result.succeeded
        assert result.accepted_code is None
        if "failure_code" in case:
            assert result.failure_code == case["failure_code"]
        return

    assert result.succeeded, result.failure_code
    assert result.accepted_code in case["exact_candidates"]
    assert result.candidate_ordinal is not None
    assert 0 <= result.candidate_ordinal < result.candidate_count


def test_every_xfail_reason_names_a_case_in_the_corpus() -> None:
    assert XFAIL_REASONS.keys() <= {case["id"] for case in CASES}


def test_both_partitions_are_represented() -> None:
    partitions = {case["partition"] for case in CASES}
    assert partitions == {"development", "holdout"}
