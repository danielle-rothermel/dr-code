from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def test_writes_length_tables_and_plots(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    output_dir = tmp_path / "analysis"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_humaneval_lengths.py"),
            "--snapshot",
            str(root / "tests" / "corpus" / "humanevalplus_snapshot.json"),
            "--output-dir",
            str(output_dir),
            "--unit",
            "bytes",
            "--bins",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Loaded 164 HumanEval tasks" in completed.stdout
    assert "Summary (bytes):" in completed.stdout

    measurements = pl.read_csv(output_dir / "humaneval_task_lengths.csv")
    assert measurements.height == 164
    assert measurements.get_column("task_id").n_unique() == 164
    assert {
        "code_without_comments_characters",
        "code_without_comments_bytes",
        "docstrings_characters",
        "docstrings_bytes",
        "hash_comments_characters",
        "hash_comments_bytes",
        "comments_and_docstrings_characters",
        "comments_and_docstrings_bytes",
    }.issubset(measurements.columns)
    assert (
        measurements.get_column("comments_and_docstrings_bytes")
        > measurements.get_column("comments_and_docstrings_characters")
    ).any()

    summary = pl.read_csv(output_dir / "humaneval_length_summary.csv")
    assert summary.height == 10
    assert set(summary.get_column("unit")) == {"characters", "bytes"}

    for filename in (
        "humaneval_length_histograms_bytes.png",
        "humaneval_length_ecdf_bytes.png",
        "humaneval_comments_vs_code_bytes.png",
    ):
        assert (output_dir / filename).stat().st_size > 0
