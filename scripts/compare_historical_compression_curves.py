#!/usr/bin/env python3

"""Compare historical and current description-compression curves."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal, cast

import matplotlib
import polars as pl
from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.metrics.compression import zstd_compressed_bytes

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_DATA_ROOT = Path.home() / "drotherm" / "data" / ".codex" / "dr-code"
_DEFAULT_OLD_RESULTS = (
    Path.home()
    / "drotherm"
    / "repos"
    / "nl_latents"
    / "results"
    / "compression_baselines"
    / "compression_baselines.parquet"
)
_DEFAULT_DICTIONARY_DIR = (
    _DATA_ROOT / "compression-dictionaries" / "mbpp-pro-4k-v1"
)
_DEFAULT_HISTORICAL_MODEL = "openrouter:openai/gpt-5-nano/low\nv1"
_SYNTHETIC_MODEL = "synthetic:openai/gpt-5-nano/low"
_ZSTD_LEVEL = 22
_PANEL_METHODS = ("zstd22", "zstd_dict22_4k")
_PANEL_TITLES = {
    "zstd22": "Zstd 22",
    "zstd_dict22_4k": "Zstd 22 + 4 KiB Dict",
}
_GT_METHODS = (
    "raw_utf8",
    "zstd22",
    "minify_raw",
    "minify_zstd22",
    "zstd_dict22_4k",
    "minify_zstd_dict22_4k",
)
_GT_LABELS = {
    "raw_utf8": "raw",
    "zstd22": "zstd",
    "minify_raw": "min",
    "minify_zstd22": "min+z",
    "zstd_dict22_4k": "dict",
    "minify_zstd_dict22_4k": "min+dict",
}
_GT_Y_POSITIONS = {
    "minify_raw": 1.02,
    "minify_zstd_dict22_4k": 1.055,
    "minify_zstd22": 1.09,
    "zstd_dict22_4k": 1.02,
    "zstd22": 1.055,
    "raw_utf8": 1.09,
}
_NEW_REQUIRED_COLUMNS = (
    "model_config_label",
    "budget_ratio",
    "max_budget",
    "task_id",
    "sample_id",
    "repeat_index",
    "generated_description",
    "test_pass_fraction",
    "gt_code_without_comments",
)
_CAVEAT = "Runs differ in tasks, date, and infrastructure."


class _DictionaryFile(FrozenModel):
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)


class _DictionaryBundleManifest(FrozenModel):
    schema_version: Literal[1] = 1
    source: str
    training_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_corpus_hash_algorithm: Literal["sha256-length-prefixed-utf8-v1"]
    requested_size: Literal[4096]
    zstandard_version: str
    python_minifier_version: str
    training_warnings: tuple[str, ...] = ()
    text: _DictionaryFile
    code: _DictionaryFile
    minified_code: _DictionaryFile


@dataclass(frozen=True, slots=True)
class _DictionaryBundle:
    manifest: _DictionaryBundleManifest
    text: bytes
    code: bytes
    minified_code: bytes


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path.resolve()


def _output_directory(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _default_output_directory() -> Path:
    now = datetime.now().astimezone()
    return (
        _DATA_ROOT
        / now.strftime("%Y-%m-%d")
        / "figs"
        / f"{now:%H%M}-historical-compression-comparison"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dictionary_file(
    directory: Path,
    descriptor: _DictionaryFile,
) -> bytes:
    filename = Path(descriptor.filename)
    if filename.name != descriptor.filename:
        raise ValueError(
            f"dictionary filename must not contain directories: "
            f"{descriptor.filename!r}"
        )
    path = directory / filename
    if not path.is_file():
        raise ValueError(f"dictionary file is missing: {path}")
    value = path.read_bytes()
    if len(value) != descriptor.byte_count:
        raise ValueError(f"dictionary byte count does not match: {path}")
    if _sha256_bytes(value) != descriptor.sha256:
        raise ValueError(f"dictionary SHA-256 does not match: {path}")
    return value


def _load_dictionary_bundle(directory: Path) -> _DictionaryBundle:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dictionary manifest is missing: {manifest_path}")
    manifest = _DictionaryBundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    return _DictionaryBundle(
        manifest=manifest,
        text=_load_dictionary_file(directory, manifest.text),
        code=_load_dictionary_file(directory, manifest.code),
        minified_code=_load_dictionary_file(directory, manifest.minified_code),
    )


def _require_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _validate_new_results(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(frame, _NEW_REQUIRED_COLUMNS, label="new results")
    selected = frame.select(
        pl.col("model_config_label").cast(pl.String),
        pl.col("budget_ratio").cast(pl.Float64, strict=True),
        pl.col("max_budget").cast(pl.Int64, strict=True),
        pl.col("task_id").cast(pl.String),
        pl.col("sample_id").cast(pl.String),
        pl.col("repeat_index").cast(pl.Int64, strict=True),
        pl.col("generated_description").cast(pl.String),
        pl.col("test_pass_fraction").cast(pl.Float64, strict=True),
        pl.col("gt_code_without_comments").cast(pl.String),
    )
    if selected.null_count().to_numpy().sum() != 0:
        raise ValueError("new results must not contain null values")
    invalid_pass_rate = selected.filter(
        ~pl.col("test_pass_fraction").is_finite()
        | ~pl.col("test_pass_fraction").is_between(0.0, 1.0, closed="both")
    )
    if invalid_pass_rate.height:
        raise ValueError(
            "test_pass_fraction must be finite and between 0 and 1"
        )
    invalid_budget_ratio = selected.filter(
        ~pl.col("budget_ratio").is_finite() | (pl.col("budget_ratio") <= 0)
    )
    if invalid_budget_ratio.height:
        raise ValueError("budget_ratio must be finite and positive")
    if selected.filter(pl.col("max_budget") < 0).height:
        raise ValueError("max_budget must be non-negative")
    if selected.filter(
        (pl.col("task_id").str.len_chars() == 0)
        | (pl.col("sample_id").str.len_chars() == 0)
        | (pl.col("gt_code_without_comments").str.len_chars() == 0)
    ).height:
        raise ValueError(
            "task_id, sample_id, and gt_code_without_comments must be non-empty"
        )
    identity = [
        "model_config_label",
        "budget_ratio",
        "task_id",
        "sample_id",
        "repeat_index",
    ]
    if selected.select(identity).is_duplicated().any():
        raise ValueError("new results contain duplicate generation identities")
    conflicting_tasks = (
        selected.group_by("task_id")
        .agg(
            pl.col("gt_code_without_comments")
            .n_unique()
            .alias("gt_code_count"),
        )
        .filter(pl.col("gt_code_count") != 1)
    )
    if conflicting_tasks.height:
        raise ValueError("each task_id must have one GT code")
    for raw_row in selected.iter_rows(named=True):
        row = cast(Mapping[str, object], raw_row)
        expected = round(
            float(cast(float, row["budget_ratio"]))
            * len(str(row["gt_code_without_comments"]))
        )
        actual = int(cast(int, row["max_budget"]))
        if actual != expected:
            raise ValueError(
                "max_budget must equal round(budget_ratio * "
                "len(gt_code_without_comments)); "
                f"task {row['task_id']!r} has {actual}, expected {expected}"
            )
    return selected


def _select_new_model(
    frame: pl.DataFrame,
    requested_model: str | None,
) -> tuple[pl.DataFrame, str]:
    models = sorted(frame.get_column("model_config_label").unique().to_list())
    if requested_model is None:
        if len(models) != 1:
            raise ValueError(
                "new results contain multiple models; pass --new-model"
            )
        requested_model = models[0]
    selected = frame.filter(pl.col("model_config_label") == requested_model)
    if selected.is_empty():
        raise ValueError(f"new model is absent: {requested_model!r}")
    return selected, requested_model


def _synthetic_new_results() -> pl.DataFrame:
    budget_ratios = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
    rows: list[dict[str, str | int | float]] = []
    for task_index in range(12):
        task_id = f"SyntheticEval/{task_index}"
        entry_point = f"solve_{task_index}"
        operations = "".join(
            f"    total += {offset}\n"
            for offset in range(1, 2 + task_index % 6)
        )
        gt_code = (
            f"def {entry_point}(value: int) -> int:\n"
            "    total = value\n"
            f"{operations}"
            "    return total\n"
        )
        for ratio_index, budget_ratio in enumerate(budget_ratios):
            max_budget = round(budget_ratio * len(gt_code))
            for repeat_index in range(2):
                description_source = (
                    f"Task {task_index} transforms an integer by adding a "
                    f"fixed offset. The implementation should preserve the "
                    f"public function name {entry_point}. "
                ) * 20
                synthetic_overage = (
                    3 if repeat_index == 1 and task_index % 5 == 0 else 0
                )
                description = description_source[
                    : max_budget + synthetic_overage
                ]
                pass_rate = min(
                    1.0,
                    0.35
                    + ratio_index * 0.11
                    + (task_index % 3) * 0.015
                    + repeat_index * 0.01,
                )
                rows.append(
                    {
                        "model_config_label": _SYNTHETIC_MODEL,
                        "budget_ratio": budget_ratio,
                        "max_budget": max_budget,
                        "task_id": task_id,
                        "sample_id": f"synthetic/{task_id}",
                        "repeat_index": repeat_index,
                        "generated_description": description,
                        "test_pass_fraction": pass_rate,
                        "gt_code_without_comments": gt_code,
                    }
                )
    return pl.DataFrame(rows)


def _historical_curve_rows(
    old: pl.DataFrame,
    *,
    model: str,
    expected_repeats: int,
) -> pl.DataFrame:
    required = (
        "row_kind",
        "model_config_label",
        "compression_method",
        "budget",
        "task_id",
        "data_sample_id",
        "source_sample_id",
        "score_bytes",
        "gt_code_raw_bytes",
        "test_pass_fraction",
    )
    _require_columns(old, required, label="historical results")
    autoencoder = old.filter(
        (pl.col("row_kind") == "autoencoder")
        & (pl.col("model_config_label") == model)
    ).with_columns(
        pl.col("budget").cast(pl.Int64, strict=False),
        pl.col("score_bytes").cast(pl.Float64, strict=False),
        pl.col("gt_code_raw_bytes").cast(pl.Float64, strict=False),
        pl.col("test_pass_fraction").cast(pl.Float64, strict=False),
    )
    if autoencoder.is_empty():
        raise ValueError(f"historical model is absent: {model!r}")
    invalid_numeric_columns = [
        column
        for column in (
            "budget",
            "score_bytes",
            "gt_code_raw_bytes",
            "test_pass_fraction",
        )
        if autoencoder.select(
            (pl.col(column).is_null() | ~pl.col(column).is_finite()).any()
        ).item()
    ]
    if invalid_numeric_columns:
        raise ValueError(
            "invalid historical numeric values: "
            + ", ".join(invalid_numeric_columns)
        )
    expected_task_count = autoencoder.get_column("task_id").n_unique()
    keys = [
        "budget",
        "task_id",
        "data_sample_id",
        "source_sample_id",
    ]
    raw = (
        autoencoder.filter(pl.col("compression_method") == "raw_utf8")
        .group_by(keys)
        .agg(pl.col("score_bytes").mean().alias("raw_score_bytes"))
    )
    frames: list[pl.DataFrame] = []
    for method in _PANEL_METHODS:
        selected = (
            autoencoder.filter(pl.col("compression_method") == method)
            .group_by(keys)
            .agg(
                pl.len().alias("source_row_count"),
                pl.col("score_bytes").mean().alias("selected_score_bytes"),
                pl.col("gt_code_raw_bytes").mean().alias("gt_code_raw_bytes"),
                pl.col("test_pass_fraction")
                .mean()
                .alias("test_pass_fraction"),
            )
        )
        paired = (
            selected.join(raw, on=keys, how="inner", validate="1:1")
            .with_columns(
                pl.min_horizontal(
                    "selected_score_bytes", "raw_score_bytes"
                ).alias("end_score_bytes")
            )
            .with_columns(
                (
                    pl.col("raw_score_bytes") / pl.col("gt_code_raw_bytes")
                ).alias("raw_ratio"),
                (
                    pl.col("end_score_bytes") / pl.col("gt_code_raw_bytes")
                ).alias("compressed_ratio"),
            )
        )
        coverage = (
            paired.group_by("budget", "task_id")
            .agg(
                pl.struct("data_sample_id", "source_sample_id")
                .n_unique()
                .alias("observed_slot_count")
            )
            .with_columns(
                pl.min_horizontal(
                    "observed_slot_count",
                    pl.lit(expected_repeats),
                ).alias("covered_slot_count")
            )
            .group_by("budget")
            .agg(pl.col("covered_slot_count").sum())
        )
        frames.append(
            paired.group_by("budget")
            .agg(
                pl.col("task_id").n_unique().alias("task_count"),
                pl.col("source_row_count").sum().alias("row_count"),
                pl.col("raw_ratio").mean(),
                pl.col("compressed_ratio").mean(),
                pl.col("test_pass_fraction").mean().alias("pass_rate"),
            )
            .join(coverage, on="budget", how="left", validate="1:1")
            .with_columns(
                pl.lit("historical").alias("run"),
                pl.lit(model).alias("model_config_label"),
                pl.lit(method).alias("panel_method"),
                pl.lit("fixed_budget").alias("treatment_kind"),
                pl.col("budget").cast(pl.Float64).alias("treatment_value"),
                pl.col("budget").cast(pl.String).alias("treatment_label"),
                pl.col("budget").alias("fixed_budget"),
                pl.lit(None, dtype=pl.Float64).alias("target_budget_ratio"),
                pl.lit(None, dtype=pl.Int64).alias("min_max_budget"),
                pl.lit(None, dtype=pl.Float64).alias("mean_max_budget"),
                pl.lit(None, dtype=pl.Int64).alias("max_max_budget"),
                pl.lit(None, dtype=pl.Int64).alias("over_budget_row_count"),
                pl.lit(None, dtype=pl.Float64).alias("over_budget_rate"),
            )
        )
    return _add_coverage(
        pl.concat(frames),
        expected_task_count=expected_task_count,
        expected_repeats=expected_repeats,
    )


def _new_curve_rows(
    new: pl.DataFrame,
    bundle: _DictionaryBundle,
    *,
    model: str,
    expected_repeats: int,
) -> pl.DataFrame:
    scored_rows: list[dict[str, str | int | float]] = []
    for raw_row in new.iter_rows(named=True):
        row = cast(Mapping[str, object], raw_row)
        description = str(row["generated_description"])
        gt_code = str(row["gt_code_without_comments"])
        raw_description = description.encode("utf-8")
        gt_raw_size = len(gt_code.encode("utf-8"))
        if gt_raw_size == 0:
            raise ValueError(f"task has empty GT code: {row['task_id']!r}")
        for method, dictionary in (
            ("zstd22", None),
            ("zstd_dict22_4k", bundle.text),
        ):
            compressed = zstd_compressed_bytes(
                raw_description,
                level=_ZSTD_LEVEL,
                dictionary=dictionary,
                compact_frame=True,
            )
            scored_rows.append(
                {
                    "model_config_label": model,
                    "target_budget_ratio": float(
                        cast(float, row["budget_ratio"])
                    ),
                    "treatment_label": (
                        f"{float(cast(float, row['budget_ratio'])):g}×"
                    ),
                    "max_budget": int(cast(int, row["max_budget"])),
                    "over_budget": int(
                        len(description) > int(cast(int, row["max_budget"]))
                    ),
                    "task_id": str(row["task_id"]),
                    "repeat_index": int(cast(int, row["repeat_index"])),
                    "panel_method": method,
                    "raw_ratio": len(raw_description) / gt_raw_size,
                    "compressed_ratio": min(
                        len(raw_description), len(compressed)
                    )
                    / gt_raw_size,
                    "pass_rate": float(cast(float, row["test_pass_fraction"])),
                }
            )
    scored = pl.DataFrame(scored_rows)
    expected_task_count = new.get_column("task_id").n_unique()
    coverage_keys = [
        "panel_method",
        "target_budget_ratio",
        "treatment_label",
    ]
    coverage = (
        scored.group_by(*coverage_keys, "task_id")
        .agg(pl.col("repeat_index").n_unique().alias("observed_slot_count"))
        .with_columns(
            pl.min_horizontal(
                "observed_slot_count",
                pl.lit(expected_repeats),
            ).alias("covered_slot_count")
        )
        .group_by(coverage_keys)
        .agg(pl.col("covered_slot_count").sum())
    )
    aggregated = (
        scored.group_by(coverage_keys)
        .agg(
            pl.col("task_id").n_unique().alias("task_count"),
            pl.len().alias("row_count"),
            pl.col("max_budget").min().alias("min_max_budget"),
            pl.col("max_budget").mean().alias("mean_max_budget"),
            pl.col("max_budget").max().alias("max_max_budget"),
            pl.col("over_budget").sum().alias("over_budget_row_count"),
            pl.col("over_budget").mean().alias("over_budget_rate"),
            pl.col("raw_ratio").mean(),
            pl.col("compressed_ratio").mean(),
            pl.col("pass_rate").mean(),
        )
        .join(coverage, on=coverage_keys, how="left", validate="1:1")
        .with_columns(
            pl.lit("current").alias("run"),
            pl.lit(model).alias("model_config_label"),
            pl.lit("target_budget_ratio").alias("treatment_kind"),
            pl.col("target_budget_ratio").alias("treatment_value"),
            pl.lit(None, dtype=pl.Int64).alias("fixed_budget"),
        )
    )
    return _add_coverage(
        aggregated,
        expected_task_count=expected_task_count,
        expected_repeats=expected_repeats,
    )


def _add_coverage(
    frame: pl.DataFrame,
    *,
    expected_task_count: int,
    expected_repeats: int,
) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.lit(expected_task_count).alias("expected_task_count"),
            pl.lit(expected_task_count * expected_repeats).alias(
                "expected_row_count"
            ),
        )
        .with_columns(
            pl.max_horizontal(
                pl.lit(0),
                pl.col("expected_row_count") - pl.col("covered_slot_count"),
            ).alias("missing_row_count")
        )
        .select(
            "run",
            "model_config_label",
            "panel_method",
            "treatment_kind",
            "treatment_value",
            "treatment_label",
            "fixed_budget",
            "target_budget_ratio",
            "task_count",
            "expected_task_count",
            "row_count",
            "covered_slot_count",
            "expected_row_count",
            "missing_row_count",
            "min_max_budget",
            "mean_max_budget",
            "max_max_budget",
            "over_budget_row_count",
            "over_budget_rate",
            "raw_ratio",
            "compressed_ratio",
            "pass_rate",
        )
        .sort("run", "panel_method", "treatment_value")
    )


def _historical_gt_references(old: pl.DataFrame) -> pl.DataFrame:
    required = (
        "row_kind",
        "payload_kind",
        "task_id",
        "compression_method",
        "score_bytes",
        "gt_code_raw_bytes",
    )
    _require_columns(old, required, label="historical results")
    gt = old.filter(
        (pl.col("row_kind") == "static_humaneval")
        & (pl.col("payload_kind") == "gt_code")
        & pl.col("compression_method").is_in(_GT_METHODS)
    ).with_columns(
        pl.col("score_bytes").cast(pl.Float64, strict=False),
        pl.col("gt_code_raw_bytes").cast(pl.Float64, strict=False),
    )
    return (
        gt.group_by("task_id", "compression_method")
        .agg(
            pl.col("score_bytes").mean(),
            pl.col("gt_code_raw_bytes").mean(),
        )
        .with_columns(
            (pl.col("score_bytes") / pl.col("gt_code_raw_bytes")).alias(
                "mean_ratio_to_gt_raw_bytes"
            )
        )
        .group_by("compression_method")
        .agg(
            pl.col("task_id").n_unique().alias("task_count"),
            pl.col("mean_ratio_to_gt_raw_bytes").mean(),
        )
        .with_columns(
            pl.lit("historical HumanEval artifact").alias("reference_source")
        )
        .select(
            "reference_source",
            "compression_method",
            "task_count",
            "mean_ratio_to_gt_raw_bytes",
        )
    )


def _plot_curves(
    curves: pl.DataFrame,
    gt_references: pl.DataFrame,
    *,
    historical_model: str,
    new_model: str,
    synthetic: bool,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True)
    run_colors = {"historical": "#777777", "current": "#0072B2"}
    for axis, method in zip(axes, _PANEL_METHODS, strict=True):
        panel = curves.filter(pl.col("panel_method") == method)
        for run in ("historical", "current"):
            selected = panel.filter(pl.col("run") == run).sort(
                "treatment_value"
            )
            treatment_labels: list[str] = selected.get_column(
                "treatment_label"
            ).to_list()
            raw_ratios: list[float] = selected.get_column(
                "raw_ratio"
            ).to_list()
            compressed_ratios: list[float] = selected.get_column(
                "compressed_ratio"
            ).to_list()
            pass_rates: list[float] = selected.get_column(
                "pass_rate"
            ).to_list()
            color = run_colors[run]
            axis.plot(
                raw_ratios,
                pass_rates,
                color=color,
                linestyle=":",
                linewidth=1.7,
                marker="o",
                markersize=5,
                markerfacecolor="white",
            )
            axis.plot(
                compressed_ratios,
                pass_rates,
                color=color,
                linestyle="-",
                linewidth=2.0,
                marker="o",
                markersize=5,
            )
            for raw_ratio, compressed_ratio, pass_rate in zip(
                raw_ratios,
                compressed_ratios,
                pass_rates,
                strict=True,
            ):
                axis.annotate(
                    "",
                    xy=(compressed_ratio, pass_rate),
                    xytext=(raw_ratio, pass_rate),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "alpha": 0.24,
                        "linewidth": 1.0,
                    },
                )
            if run == "current":
                for treatment_label, x_value, y_value in zip(
                    treatment_labels,
                    compressed_ratios,
                    pass_rates,
                    strict=True,
                ):
                    axis.annotate(
                        treatment_label,
                        (x_value, y_value),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                    )
        _plot_gt_references(axis, gt_references)
        axis.set_xscale("log")
        axis.set_title(_PANEL_TITLES[method])
        axis.set_xlabel("Description / GT Code (bytes)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Pass Rate")
    axes[0].set_ylim(0, 1.14)
    figure.text(
        0.5,
        0.975,
        f"{_display_model(historical_model)} vs "
        f"{_display_model(new_model)}: Compression vs Pass Rate",
        ha="center",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.94,
        "Stars: GT code  ·  Labels: target budget ratio",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=run_colors["historical"],
                marker="o",
                linewidth=2,
                label=f"Historical · {_display_model(historical_model)}",
            ),
            Line2D(
                [],
                [],
                color=run_colors["current"],
                marker="o",
                linewidth=2,
                label=f"Current · {_display_model(new_model)}",
            ),
        ],
        title="Experiment",
        loc="lower center",
        bbox_to_anchor=(0.32, 0.065),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#333333",
                linestyle=":",
                marker="o",
                markerfacecolor="white",
                linewidth=1.7,
                label="Off",
            ),
            Line2D(
                [],
                [],
                color="#333333",
                linestyle="-",
                marker="o",
                linewidth=2,
                label="On",
            ),
        ],
        title="Lossless Compression",
        loc="lower center",
        bbox_to_anchor=(0.68, 0.065),
        ncol=2,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    if synthetic:
        figure.text(
            0.5,
            0.5,
            "SYNTHETIC",
            ha="center",
            va="center",
            fontsize=42,
            color="#B2182B",
            alpha=0.14,
            rotation=18,
        )
    figure.tight_layout(rect=(0, 0.18, 1, 0.90))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_gt_references(axis: Axes, references: pl.DataFrame) -> None:
    colors = plt.colormaps["tab10"](
        [index / len(_GT_METHODS) for index in range(len(_GT_METHODS))]
    )
    for method, color in zip(_GT_METHODS, colors, strict=True):
        method_rows = references.filter(pl.col("compression_method") == method)
        if method_rows.is_empty():
            continue
        x_value = float(method_rows.item(0, "mean_ratio_to_gt_raw_bytes"))
        y_position = _GT_Y_POSITIONS[method]
        axis.scatter(
            [x_value],
            [y_position],
            marker="*",
            s=100,
            color=[color],
            edgecolors="white",
            linewidths=0.5,
        )
        axis.annotate(
            _GT_LABELS[method],
            (x_value, y_position),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
        )


def _display_model(value: str) -> str:
    return value.replace("\n", " ").removesuffix(" v1")


def _treatment_summary(row: Mapping[str, object]) -> str:
    if row["treatment_kind"] == "fixed_budget":
        return f"fixed_budget={int(cast(int, row['fixed_budget']))}"
    return (
        f"target_ratio={float(cast(float, row['target_budget_ratio'])):g}×, "
        f"max_budget={int(cast(int, row['min_max_budget']))}-"
        f"{int(cast(int, row['max_max_budget']))} chars "
        f"(mean={float(cast(float, row['mean_max_budget'])):.1f}), "
        f"over_budget={int(cast(int, row['over_budget_row_count']))}/"
        f"{int(cast(int, row['row_count']))}"
    )


def _summary_lines(curves: pl.DataFrame) -> list[str]:
    lines: list[str] = []
    for run in ("historical", "current"):
        run_rows = curves.filter(pl.col("run") == run)
        model = str(run_rows.item(0, "model_config_label"))
        lines.append(f"{run.title()} model: {_display_model(model)}")
        for raw_row in run_rows.sort(
            "panel_method", "treatment_value"
        ).iter_rows(named=True):
            row = cast(Mapping[str, object], raw_row)
            lines.append(
                f"  {row['panel_method']} {_treatment_summary(row)}: "
                f"raw={float(cast(float, row['raw_ratio'])):.6f}, "
                f"compressed="
                f"{float(cast(float, row['compressed_ratio'])):.6f}, "
                f"pass={float(cast(float, row['pass_rate'])):.6f}, "
                f"coverage={int(cast(int, row['covered_slot_count']))}/"
                f"{int(cast(int, row['expected_row_count']))}"
            )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce historical description-compression calculations and "
            "compare them with current experiment rows."
        )
    )
    parser.add_argument(
        "--old-results",
        type=_existing_file,
        default=_DEFAULT_OLD_RESULTS,
        help=f"historical nl_latents parquet (default: {_DEFAULT_OLD_RESULTS})",
    )
    new_group = parser.add_mutually_exclusive_group(required=True)
    new_group.add_argument(
        "--new-results",
        type=_existing_file,
        help="current results parquet using the canonical input columns",
    )
    new_group.add_argument(
        "--synthetic",
        action="store_true",
        help="use deterministic synthetic current rows and watermark the plot",
    )
    parser.add_argument(
        "--dictionary-dir",
        type=_existing_directory,
        default=_DEFAULT_DICTIONARY_DIR,
        help=f"frozen dictionary bundle (default: {_DEFAULT_DICTIONARY_DIR})",
    )
    parser.add_argument(
        "--historical-model",
        default=_DEFAULT_HISTORICAL_MODEL,
        help="exact historical model_config_label",
    )
    parser.add_argument(
        "--new-model",
        help="current model label; required only when input contains many",
    )
    parser.add_argument(
        "--historical-expected-repeats",
        type=int,
        default=5,
    )
    parser.add_argument("--new-expected-repeats", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=_output_directory,
        help=f"output directory under {_DATA_ROOT} by default",
    )
    arguments = parser.parse_args()
    if arguments.historical_expected_repeats <= 0:
        parser.error("--historical-expected-repeats must be positive")
    if arguments.new_expected_repeats <= 0:
        parser.error("--new-expected-repeats must be positive")

    output_dir = arguments.output_dir or _default_output_directory()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_dictionary_bundle(arguments.dictionary_dir)
    old = pl.read_parquet(arguments.old_results)
    if arguments.synthetic:
        new = _synthetic_new_results()
        synthetic_input_path = output_dir / "synthetic_new_results.parquet"
        new.write_parquet(synthetic_input_path)
    else:
        new = pl.read_parquet(arguments.new_results)
        synthetic_input_path = None
    new = _validate_new_results(new)
    new, new_model = _select_new_model(new, arguments.new_model)

    historical_curves = _historical_curve_rows(
        old,
        model=arguments.historical_model,
        expected_repeats=arguments.historical_expected_repeats,
    )
    current_curves = _new_curve_rows(
        new,
        bundle,
        model=new_model,
        expected_repeats=arguments.new_expected_repeats,
    )
    curves = pl.concat([historical_curves, current_curves]).sort(
        "run", "panel_method", "treatment_value"
    )
    gt_references = _historical_gt_references(old).sort("compression_method")

    curves_path = output_dir / "historical_current_compression_curves.csv"
    references_path = output_dir / "fixed_gt_compression_references.csv"
    plot_path = output_dir / "historical_current_compression_comparison.png"
    metadata_path = output_dir / "historical_current_compression_metadata.json"
    log_path = output_dir / "historical_current_compression_comparison.log"
    curves.write_csv(curves_path)
    gt_references.write_csv(references_path)
    _plot_curves(
        curves,
        gt_references,
        historical_model=arguments.historical_model,
        new_model=new_model,
        synthetic=arguments.synthetic,
        path=plot_path,
    )
    metadata = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "descriptive_comparison_only": True,
        "caveat": _CAVEAT,
        "synthetic_new_data": bool(arguments.synthetic),
        "old_results": str(arguments.old_results),
        "old_results_sha256": _sha256_file(arguments.old_results),
        "new_results": (
            str(synthetic_input_path)
            if synthetic_input_path is not None
            else str(arguments.new_results)
        ),
        "new_results_sha256": _sha256_file(
            synthetic_input_path
            if synthetic_input_path is not None
            else arguments.new_results
        ),
        "new_results_columns": list(_NEW_REQUIRED_COLUMNS),
        "historical_model": arguments.historical_model,
        "new_model": new_model,
        "historical_expected_repeats": arguments.historical_expected_repeats,
        "new_expected_repeats": arguments.new_expected_repeats,
        "current_treatment": {
            "kind": "target_character_budget_ratio",
            "derivation": (
                "max_budget = round(budget_ratio * "
                "len(gt_code_without_comments))"
            ),
            "grouping": "budget_ratio",
            "plot_coordinate": (
                "observed mean description bytes / raw GT-code bytes"
            ),
        },
        "gt_compression_references": {
            "fixed": True,
            "source": "historical HumanEval artifact",
            "current_run_recomputed": False,
        },
        "aggregation": (
            "Mean per observed generation row after pairing raw and selected "
            "compression rows; compressed bytes use min(raw, compressed)."
        ),
        "coverage": {
            "expected_task_population": (
                "All task_id values observed for the selected model across "
                "treatments."
            ),
            "expected_rows_per_treatment": (
                "expected_task_count * configured expected repeats"
            ),
            "covered_slots": (
                "Distinct paired task/repeat slots, capped at the configured "
                "repeat count per task."
            ),
        },
        "runtime_versions": {
            "python-minifier": version("python-minifier"),
            "zstandard": version("zstandard"),
        },
        "dictionary_bundle": bundle.manifest.model_dump(mode="json"),
        "dictionary_dir": str(arguments.dictionary_dir),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    report = [
        *_summary_lines(curves),
        f"Caveat: {_CAVEAT}",
        f"Wrote plotted curve points: {curves_path}",
        f"Wrote fixed GT reference points: {references_path}",
        f"Wrote comparison plot: {plot_path}",
        f"Wrote metadata: {metadata_path}",
        f"Wrote output log: {log_path}",
    ]
    output = "\n".join(report) + "\n"
    log_path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
