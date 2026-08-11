from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_help_lists_all_supported_datasets() -> None:
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_generation_corpus.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "{human_eval,mbpp_pro,humaneval_pro,class_eval,"
        "bigcodebench_lite_pro,nl_latents}"
    ) in completed.stdout
    assert "without executing any candidate code" in completed.stdout
