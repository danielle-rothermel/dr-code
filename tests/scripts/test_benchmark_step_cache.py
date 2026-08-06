from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def test_reports_candidate_and_execution_request_reuse(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    parquet = tmp_path / "generations.parquet"
    source = "def has_close_elements(numbers, threshold):\n    return False\n"
    pl.DataFrame(
        {
            "sample_id": ["sample-1", "sample-2"],
            "task_id": ["HumanEval/148", "HumanEval/148"],
            "decoder_output": [source, source],
        }
    ).write_parquet(parquet)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_step_cache.py"),
            "HumanEval/148",
            "--parquet",
            str(parquet),
            "--humaneval-snapshot",
            str(root / "tests" / "corpus" / "humanevalplus_snapshot.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Raw extracted-source hit rate" in completed.stdout
    assert (
        "Final postprocessed-source hit rate: 50.00% "
        "(1/2 hits; 1 unique)" in completed.stdout
    )
    assert (
        "Test-safe execution-request hit rate: 50.00% "
        "(1/2 hits; 1 unique)" in completed.stdout
    )
    assert "No candidate code was executed." in completed.stdout


def test_bootstraps_source_and_test_safe_reuse_across_tasks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    parquet = tmp_path / "generations.parquet"
    source = "def candidate(value):\n    return value\n"
    pl.DataFrame(
        {
            "sample_id": ["zero-1", "zero-2", "one-1", "one-2"],
            "task_id": [
                "HumanEval/0",
                "HumanEval/0",
                "HumanEval/1",
                "HumanEval/1",
            ],
            "decoder_output": [source, source, source, source],
        }
    ).write_parquet(parquet)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_step_cache.py"),
            "--task-count",
            "2",
            "--bootstrap-resamples",
            "100",
            "--parquet",
            str(parquet),
            "--humaneval-snapshot",
            str(root / "tests" / "corpus" / "humanevalplus_snapshot.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "Aggregate final postprocessed-source hit rate: 75.00%"
        in completed.stdout
    )
    assert (
        "Aggregate test-safe execution-request hit rate: 50.00%"
        in completed.stdout
    )
    assert (
        "Candidate reuse was planned without executing candidate code."
        in completed.stdout
    )
