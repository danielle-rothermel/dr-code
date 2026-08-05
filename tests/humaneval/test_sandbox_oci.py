from __future__ import annotations

import os

import pytest

from dr_code.core.execution.sandbox import (
    MAX_SANDBOX_OUTPUT_BYTES,
    run_python_in_sandbox,
)
from dr_code.humaneval.runner import evaluate_humaneval_code
from dr_code.humaneval.scoring import (
    CompletedScore,
    score_humaneval_submission,
)
from dr_code.humaneval.task import HumanEvalTask


# OCI probes run in CI; local runs require explicit opt-in.
pytestmark = [
    pytest.mark.oci,
    pytest.mark.skipif(
        os.environ.get("DR_CODE_RUN_SANDBOX_TESTS") != "1"
        and os.environ.get("CI") is None,
        reason="real OCI sandbox probes require DR_CODE_RUN_SANDBOX_TESTS=1",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _warm_sandbox_container() -> None:
    # Timeouts are watchdogs; warm once so cold start is not the outcome.
    run_python_in_sandbox(
        source="input()\nprint('[]')\n",
        input_json="{}",
        timeout_seconds=30.0,
    )


def _task() -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/sandbox-fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,), (2,)]\n"
            "    results = [2, 3]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )


def test_known_good_submission_scores_inside_real_sandbox() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
    )

    assert isinstance(result, CompletedScore)
    assert result.score == 1.0


def test_candidate_sys_exit_is_scored_not_harness_failure() -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "import sys\nsys.exit(0)\ndef add_one(x):\n    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}


def test_memory_exhaustion_is_scored_not_harness_failure() -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    data = bytearray(512 * 1024 * 1024)\n"
            "    return x + 1 + data[0]\n"
        ),
        timeout_seconds=5.0,
    )

    assert result.passed is False
    assert set(result.status_counts) <= {"error", "timeout"}


def test_output_flood_is_scored_not_harness_failure() -> None:
    result = evaluate_humaneval_code(
        task=_task(),
        candidate_code=(
            "import sys\n"
            "def add_one(x):\n"
            f"    sys.stdout.write('x' * {MAX_SANDBOX_OUTPUT_BYTES + 1})\n"
            "    return x + 1\n"
        ),
        timeout_seconds=5.0,
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}
