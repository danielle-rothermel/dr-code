"""Import-order regressions for the HumanEval metric operator."""

from __future__ import annotations

import subprocess
import sys


def test_code_test_imports_in_clean_interpreter() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from dr_code.humaneval.metric_operator import CodeTest",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
