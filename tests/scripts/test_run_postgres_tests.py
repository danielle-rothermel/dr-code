from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_postgres_tests.sh"


def test_run_postgres_tests_requires_dr_store_root() -> None:
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "DR_STORE_ROOT"
    }
    completed = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "DR_STORE_ROOT is not set" in completed.stderr


def test_run_postgres_tests_rejects_a_missing_scratch_script() -> None:
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if key != "DR_STORE_ROOT"
        },
        "DR_STORE_ROOT": "/nonexistent/dr-store",
    }
    completed = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "Expected an executable scratch-server script" in completed.stderr
