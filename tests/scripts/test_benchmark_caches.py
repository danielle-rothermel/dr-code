from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def test_reports_existing_preprocessing_cache_and_candidate_reuse(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]
    parquet = tmp_path / "generations.parquet"
    source = "def has_close_elements(numbers, threshold):\n    return False\n"
    partial_hit_source = f"""```python
{source}```

```python
def other(value):
    return value
```"""
    pl.DataFrame(
        {
            "sample_id": [
                "zero-hit",
                "partial-hit",
                "fully-cached",
                "failed",
            ],
            "task_id": ["HumanEval/148"] * 4,
            "decoder_output": [
                source,
                partial_hit_source,
                source,
                "not python",
            ],
        }
    ).write_parquet(parquet)

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_caches.py"),
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

    assert "Cold SQLite whole-trace cache" in completed.stdout
    assert "Warm SQLite whole-trace cache" in completed.stdout
    assert (
        "Trace equivalence: exact for cold and warm whole-trace cache runs"
        in completed.stdout
    )
    assert "Raw extracted-source hit rate" in completed.stdout
    assert (
        "Testable candidate cache hit rate: 50.00% (2/4)" in completed.stdout
    )
    assert (
        "Overall full-test skip rate: 25.00% (1/4 nonblank rows)"
        in completed.stdout
    )
    assert (
        "Conditional full-test skip rate: 33.33% "
        "(1/3 successful rows)" in completed.stdout
    )
    assert "preprocessing failed: 1 (25.00%)" in completed.stdout
    assert "successful, zero cache hits: 1 (25.00%)" in completed.stdout
    assert "successful, partial cache hits: 1 (25.00%)" in completed.stdout
    assert "successful, fully cached: 1 (25.00%)" in completed.stdout
    assert (
        "Uncached candidates per successful row: "
        "total=2, mean=0.67, median=1.00, p95=1" in completed.stdout
    )
    assert (
        "no_candidate_survived_filtering: 1 (100.00% of failures)"
        in completed.stdout
    )
    assert (
        "Final postprocessed source-only hit rate: 50.00% "
        "(2/4 hits; 2 unique)" in completed.stdout
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
            str(root / "scripts" / "benchmark_caches.py"),
            "--task-count",
            "2",
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
        "Aggregate final postprocessed source-only hit rate: 75.00%"
        in completed.stdout
    )
    assert (
        "Aggregate uncached-to-cold whole-trace-cache speedup"
        in completed.stdout
    )
    assert (
        "Aggregate uncached-to-warm whole-trace-cache speedup"
        in completed.stdout
    )
    assert (
        "Aggregate testable candidate cache hit rate: 50.00%"
        in completed.stdout
    )
    assert "Aggregate overall full-test skip rate: 50.00%" in completed.stdout
    assert (
        "Aggregate conditional full-test skip rate: 50.00%" in completed.stdout
    )
    assert "Aggregate execution-request hit rate: 50.00%" in completed.stdout
    assert (
        "Candidate reuse was planned without executing candidate code."
        in completed.stdout
    )
