from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def test_reports_preprocessing_success_and_failures(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    script = root / "scripts" / "analyze_preprocessing_success.py"
    parquet = tmp_path / "generations.parquet"
    source = "def has_close_elements(numbers, threshold):\n    return False\n"
    multiple_candidates = f"""```python
{source}```

```python
def other(value):
    return value
```"""
    pl.DataFrame(
        {
            "sample_id": ["first", "multiple", "repeat", "failed"],
            "task_id": ["HumanEval/148"] * 4,
            "decoder_output": [
                source,
                multiple_candidates,
                source,
                "not python",
            ],
        }
    ).write_parquet(parquet)

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "HumanEval/148",
            "--parquet",
            str(parquet),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Preprocessing success: 75.00% (3/4 nonblank samples)" in (
        completed.stdout
    )
    assert "Full preprocessing:" in completed.stdout
    assert "no_candidate_survived_filtering: 1 (100.00% of failures)" in (
        completed.stdout
    )
    assert "cache" not in script.read_text().lower()


def test_bootstraps_preprocessing_success_across_tasks(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    parquet = tmp_path / "generations.parquet"
    source = "def candidate(value):\n    return value\n"
    pl.DataFrame(
        {
            "sample_id": ["zero-1", "one-1", "one-2", "one-3"],
            "task_id": [
                "HumanEval/0",
                "HumanEval/1",
                "HumanEval/1",
                "HumanEval/1",
            ],
            "decoder_output": [
                source,
                source,
                "not python",
                "still not python",
            ],
        }
    ).write_parquet(parquet)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_preprocessing_success.py"),
            "--task-count",
            "2",
            "--bootstrap-resamples",
            "100",
            "--parquet",
            str(parquet),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Aggregate preprocessing success: 50.00% (2/4" in (completed.stdout)
    assert "HumanEval/0: none" in completed.stdout
    assert "HumanEval/1: no_candidate_survived_filtering=2" in completed.stdout
    assert (
        "Aggregate preprocessing success rate: 50.00% "
        "(95% bootstrap CI 33.33% to 100.00%)" in completed.stdout
    )
    assert "Bootstrap unit: task; resamples: 100; seed: 0" in completed.stdout
