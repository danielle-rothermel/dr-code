from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl


def test_writes_long_form_compression_analysis(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    output_dir = tmp_path / "compression"
    slug = "gzip6-zstd3"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_humaneval_compression.py"),
            "--snapshot",
            str(root / "tests" / "corpus" / "humanevalplus_snapshot.json"),
            "--comp",
            "gzip:6,zstd:3",
            "--order-by",
            "gzip:6",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Loaded 164 HumanEval tasks" in completed.stdout
    assert "Compression configurations: gzip:6, zstd:3" in completed.stdout
    assert "Task ordering: gzip:6 ratio descending" in completed.stdout

    measurements = pl.read_csv(
        output_dir / f"humaneval_compression_measurements_bytes_{slug}.csv"
    )
    assert measurements.height == 164 * 2 * 2
    assert measurements.get_column("task_id").n_unique() == 164
    assert set(measurements.get_column("representation")) == {
        "code_without_comments",
        "comments_and_docstrings",
    }
    assert set(measurements.get_column("configuration")) == {
        "gzip:6",
        "zstd:3",
    }
    assert measurements.get_column("compression_ratio").is_not_null().all()
    assert (
        measurements.get_column("percent_reduction")
        == (1 - measurements.get_column("compression_ratio")) * 100
    ).all()

    summary = pl.read_csv(
        output_dir / f"humaneval_compression_summary_bytes_{slug}.csv"
    )
    assert summary.height == 2 * 2

    task_order = pl.read_csv(
        output_dir / f"humaneval_code_task_order_by_gzip6_{slug}.csv"
    )
    assert task_order.height == 164
    assert task_order.get_column("task_rank").to_list() == list(range(1, 165))
    assert task_order.get_column("task_id").n_unique() == 164
    assert task_order.get_column(
        "order_by_configuration"
    ).unique().to_list() == ["gzip:6"]
    ratios = task_order.get_column("order_by_compression_ratio")
    assert ratios.to_list() == ratios.sort(descending=True).to_list()

    for filename in (
        f"humaneval_code_compression_bytes_{slug}.png",
        f"humaneval_comments_docstrings_compression_bytes_{slug}.png",
        "humaneval_code_raw_bytes_vs_compression_ratio_gzip6.png",
        "humaneval_code_raw_bytes_vs_compression_ratio_zstd3.png",
        f"humaneval_code_raw_bytes_vs_compression_ratio_faceted_{slug}.png",
        "humaneval_code_raw_bytes_vs_bytes_saved_gzip6.png",
        "humaneval_code_raw_bytes_vs_bytes_saved_zstd3.png",
        f"humaneval_code_raw_bytes_vs_bytes_saved_faceted_{slug}.png",
        f"humaneval_code_ordered_by_gzip6_compression_ratio_{slug}.png",
        f"humaneval_code_ordered_by_gzip6_bytes_saved_{slug}.png",
    ):
        assert (output_dir / filename).stat().st_size > 0

    log_path = output_dir / f"humaneval_compression_analysis_bytes_{slug}.log"
    assert log_path.read_text(encoding="utf-8") == completed.stdout


def test_rejects_duplicate_compression_configuration(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_humaneval_compression.py"),
            "--comp",
            "gzip:6,gzip:6",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "duplicate compression configuration" in completed.stderr


def test_rejects_ordering_configuration_not_in_comparison(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2]

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_humaneval_compression.py"),
            "--comp",
            "gzip:6",
            "--order-by",
            "zstd:3",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--order-by 'zstd:3' is not present in --comp" in completed.stderr
