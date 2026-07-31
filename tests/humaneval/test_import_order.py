"""Fresh-process import contracts for the HumanEval public facade."""

from __future__ import annotations

import subprocess
import sys

import pytest


_SCORING_FACADE_IMPORTS = (
    "from dr_code.humaneval import (",
    "    CompletedScore,",
    "    HarnessFailure,",
    "    HarnessFailureCause,",
    "    HumanEvalSubmissionScore,",
    "    SubmissionOutcome,",
    "    evaluation_aggregate_metrics,",
    "    score_humaneval_submission,",
    ")",
)


@pytest.mark.parametrize(
    "imports",
    (
        "import dr_code.preprocessing",
        "\n".join(
            (
                "import dr_code.preprocessing",
                *_SCORING_FACADE_IMPORTS,
            )
        ),
        "\n".join(
            (
                *_SCORING_FACADE_IMPORTS,
                "import dr_code.preprocessing",
            )
        ),
    ),
)
def test_preprocessing_and_humaneval_facade_import_in_any_order(
    imports: str,
) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-c", imports],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
