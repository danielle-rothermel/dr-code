from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import zstandard

from dr_code.metrics.compression import train_zstd_dictionary


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_dictionary_bundle(directory: Path) -> None:
    directory.mkdir()
    text_samples = [
        (
            f"Task {index} requires transforming an integer while preserving "
            "the documented public behavior and returning the result. "
        ).encode()
        * 8
        for index in range(80)
    ]
    code_samples = [
        (
            f"def solve_{index}(value: int) -> int:\n"
            f'    """Solve task {index}."""\n'
            f"    adjusted = value + {index}\n"
            "    return adjusted\n"
        ).encode()
        * 4
        for index in range(80)
    ]
    minified_samples = [
        f"def solve_{index}(A):return A+{index}\n".encode() * 8
        for index in range(80)
    ]
    values = {
        "text": train_zstd_dictionary(text_samples, dictionary_size=4096),
        "code": train_zstd_dictionary(code_samples, dictionary_size=4096),
        "minified_code": train_zstd_dictionary(
            minified_samples,
            dictionary_size=4096,
        ),
    }
    files: dict[str, dict[str, str | int]] = {}
    for name, value in values.items():
        filename = f"{name}.zdict"
        (directory / filename).write_bytes(value)
        files[name] = {
            "filename": filename,
            "sha256": _sha256(value),
            "byte_count": len(value),
        }
    manifest = {
        "schema_version": 1,
        "source": "deterministic test corpus",
        "training_corpus_sha256": "0" * 64,
        "training_corpus_hash_algorithm": "sha256-length-prefixed-utf8-v1",
        "requested_size": 4096,
        "zstandard_version": zstandard.__version__,
        "python_minifier_version": "3.2.0",
        "training_warnings": [],
        **files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def _write_historical_results(path: Path, model: str) -> None:
    rows: list[dict[str, str | int | float]] = []
    for task_index in range(3):
        task_id = f"HistoricalEval/{task_index}"
        gt_size = 100 + task_index * 10
        for method, ratio in (
            ("raw_utf8", 1.0),
            ("zstd22", 0.72),
            ("minify_raw", 0.68),
            ("minify_zstd22", 0.61),
            ("zstd_dict22_4k", 0.49),
            ("minify_zstd_dict22_4k", 0.43),
        ):
            rows.append(
                {
                    "row_kind": "static_humaneval",
                    "payload_kind": "gt_code",
                    "model_config_label": "",
                    "compression_method": method,
                    "budget": 0,
                    "task_id": task_id,
                    "data_sample_id": "",
                    "source_sample_id": "",
                    "score_bytes": gt_size * ratio,
                    "gt_code_raw_bytes": gt_size,
                    "test_pass_fraction": 1.0,
                }
            )
        for budget_index, budget in enumerate((32, 64)):
            for repeat_index in range(2):
                raw_size = 30 + budget_index * 20 + task_index
                pass_rate = 0.4 + budget_index * 0.2 + task_index * 0.01
                for method, factor in (
                    ("raw_utf8", 1.0),
                    ("zstd22", 0.82),
                    ("zstd_dict22_4k", 0.66),
                ):
                    rows.append(
                        {
                            "row_kind": "autoencoder",
                            "payload_kind": "generated_description",
                            "model_config_label": model,
                            "compression_method": method,
                            "budget": budget,
                            "task_id": task_id,
                            "data_sample_id": f"sample/{task_id}",
                            "source_sample_id": str(repeat_index),
                            "score_bytes": raw_size * factor,
                            "gt_code_raw_bytes": gt_size,
                            "test_pass_fraction": pass_rate,
                        }
                    )
    pl.DataFrame(rows).write_parquet(path)


def _write_current_results(path: Path, model: str) -> None:
    rows: list[dict[str, str | int | float]] = []
    for task_index in range(4):
        task_id = f"CurrentEval/{task_index}"
        entry_point = f"solve_{task_index}"
        gt_code = (
            f"def {entry_point}(value: int) -> int:\n"
            f'    """Return the result for task {task_index}."""\n'
            f"    return value + {task_index}\n"
        )
        for budget_index, budget in enumerate((32, 64)):
            for repeat_index in range(2):
                rows.append(
                    {
                        "model_config_label": model,
                        "budget": budget,
                        "task_id": task_id,
                        "sample_id": f"sample/{task_id}",
                        "repeat_index": repeat_index,
                        "generated_description": (
                            f"Task {task_index} adds a stable integer offset. "
                            * (budget_index + 2)
                        ),
                        "test_pass_fraction": (
                            0.6 + budget_index * 0.2 + task_index * 0.01
                        ),
                        "gt_code": gt_code,
                        "entry_point": entry_point,
                    }
                )
    pl.DataFrame(rows).write_parquet(path)


def _command(
    *,
    old_results: Path,
    dictionary_dir: Path,
    output_dir: Path,
) -> list[str]:
    root = Path(__file__).parents[2]
    return [
        sys.executable,
        str(root / "scripts" / "compare_historical_compression_curves.py"),
        "--old-results",
        str(old_results),
        "--dictionary-dir",
        str(dictionary_dir),
        "--historical-model",
        "historical-model",
        "--historical-expected-repeats",
        "2",
        "--new-expected-repeats",
        "2",
        "--output-dir",
        str(output_dir),
    ]


def test_compares_historical_and_current_parquet_results(
    tmp_path: Path,
) -> None:
    old_results = tmp_path / "old.parquet"
    new_results = tmp_path / "new.parquet"
    dictionary_dir = tmp_path / "dictionaries"
    output_dir = tmp_path / "comparison"
    _write_historical_results(old_results, "historical-model")
    _write_current_results(new_results, "current-model")
    _write_dictionary_bundle(dictionary_dir)

    completed = subprocess.run(
        [
            *_command(
                old_results=old_results,
                dictionary_dir=dictionary_dir,
                output_dir=output_dir,
            ),
            "--new-results",
            str(new_results),
            "--new-model",
            "current-model",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    curves = pl.read_csv(
        output_dir / "historical_current_compression_curves.csv"
    )
    assert curves.height == 2 * 2 * 2
    assert set(curves.get_column("run")) == {"historical", "current"}
    assert set(curves.get_column("panel_method")) == {
        "zstd22",
        "zstd_dict22_4k",
    }
    assert curves.filter(pl.col("run") == "historical").get_column(
        "task_count"
    ).unique().to_list() == [3]
    assert curves.filter(pl.col("run") == "current").get_column(
        "task_count"
    ).unique().to_list() == [4]
    assert curves.get_column("missing_row_count").eq(0).all()

    references = pl.read_csv(
        output_dir / "historical_current_gt_references.csv"
    )
    assert references.height == 2 * 6
    assert (
        output_dir / "historical_current_compression_comparison.png"
    ).stat().st_size > 0
    metadata = json.loads(
        (
            output_dir / "historical_current_compression_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["synthetic_new_data"] is False
    assert metadata["descriptive_comparison_only"] is True
    assert (
        output_dir / "historical_current_compression_comparison.log"
    ).read_text(encoding="utf-8") == completed.stdout


def test_synthetic_mode_writes_watermarked_input(tmp_path: Path) -> None:
    old_results = tmp_path / "old.parquet"
    dictionary_dir = tmp_path / "dictionaries"
    output_dir = tmp_path / "synthetic-comparison"
    _write_historical_results(old_results, "historical-model")
    _write_dictionary_bundle(dictionary_dir)

    subprocess.run(
        [
            *_command(
                old_results=old_results,
                dictionary_dir=dictionary_dir,
                output_dir=output_dir,
            ),
            "--synthetic",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    synthetic = pl.read_parquet(output_dir / "synthetic_new_results.parquet")
    assert synthetic.columns == [
        "model_config_label",
        "budget",
        "task_id",
        "sample_id",
        "repeat_index",
        "generated_description",
        "test_pass_fraction",
        "gt_code",
        "entry_point",
    ]
    assert synthetic.height == 12 * 6 * 2
    metadata = json.loads(
        (
            output_dir / "historical_current_compression_metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["synthetic_new_data"] is True
