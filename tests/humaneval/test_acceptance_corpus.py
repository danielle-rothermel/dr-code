"""Extraction pinned against a corpus of recorded decoder outputs.

``corpus/hard_examples.json`` holds 130 real LLM decoder outputs, each
carrying a human verdict on what extraction owes it: ``candidates`` with the
exact source that should be accepted, or ``absent`` when the response
contains no answer to extract. The corpus is partitioned into a development
set and a holdout set, and the partition is visible in every test id.

These cases are evidence, not a specification. A response is recorded output
plus a human reading of it, so a case that disagrees with the pipeline is a
finding to adjudicate, never a mandate to change the contract. Disagreements
are marked ``xfail`` individually in ``XFAIL_REASONS``, one stated reason
each, so the set of open disagreements stays enumerable and a case that
starts agreeing fails loudly as ``XPASS``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest

from dr_code.humaneval.acceptance import (
    extract_humaneval_code,
    humaneval_runner,
)

_FIXTURE: Final[Path] = Path(__file__).parent / "corpus" / "hard_examples.json"

#: Case id -> why extraction does not yet agree with the recorded verdict.
#: Every entry is a deliberate open decision, not a defect to fix in place.
XFAIL_REASONS: Final[dict[str, str]] = {
    "annotation-1b961da8b9d59f41": (
        "lambda-only solution: rendering a bare lambda as a named function "
        "is an open decision, so no top-level function survives"
    ),
    "annotation-1f427a1043acf297": (
        "lambda-only solution: rendering a bare lambda as a named function "
        "is an open decision, so no top-level function survives"
    ),
    "annotation-72944a2187a55d01": (
        "singleton string container: reading the sole string out of a list "
        "literal is an open decision, so the container never compiles"
    ),
    "annotation-b0934e569fa9de7d": (
        "JSON envelope truncated at EOF: completing an unterminated "
        "envelope is an open decision, and decoding stays strict"
    ),
    "annotation-d5ed37d4dadbbc75": (
        "JSON envelope truncated at EOF: completing an unterminated "
        "envelope is an open decision, and decoding stays strict"
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
    """One param per case, xfailing the enumerated disagreements."""
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
    """One binding reused across the corpus."""
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
    """A stale reason would silently stop marking anything."""
    assert XFAIL_REASONS.keys() <= {case["id"] for case in CASES}


def test_both_partitions_are_represented() -> None:
    partitions = {case["partition"] for case in CASES}
    assert partitions == {"development", "holdout"}
