#!/usr/bin/env python3

"""Measure compression of normalized HumanEval code and comments."""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib
import polars as pl
from pydantic import ValidationError

from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.sampling import load_humaneval_rows
from dr_code.metrics.compression import (
    CompressionMethod,
    GzipConfig,
    ZstdConfig,
    compressed_bytes,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

_ROOT = Path(__file__).parents[1]
_DEFAULT_SNAPSHOT = _ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
_DATA_ROOT = Path.home() / "drotherm" / "data" / ".codex" / "dr-code"
_DEFAULT_COMPRESSIONS = "gzip:6,gzip:9,zstd:3,zstd:9"
_DEFAULT_ORDERING = "zstd:3"


@dataclass(frozen=True, slots=True)
class _Representation:
    key: str
    label: str


@dataclass(frozen=True, slots=True)
class _ComparisonMetric:
    key: str
    label: str
    filename_slug: str
    reference_value: float
    reference_label: str


@dataclass(frozen=True, slots=True)
class _EffectCurve:
    label: str
    values: tuple[float, ...]
    color: str
    line_style: str


_REPRESENTATIONS = (
    _Representation(
        "code_without_comments",
        "Normalized code without comments or docstrings",
    ),
    _Representation(
        "comments_and_docstrings",
        "Normalized comments and docstrings",
    ),
)

_COMPARISON_METRICS = (
    _ComparisonMetric(
        key="compression_ratio",
        label="Compressed bytes / raw bytes",
        filename_slug="compression_ratio",
        reference_value=1.0,
        reference_label="No size change",
    ),
    _ComparisonMetric(
        key="bytes_saved",
        label="Bytes saved per task",
        filename_slug="bytes_saved",
        reference_value=0.0,
        reference_label="No size change",
    ),
)

_ALGORITHM_COLORS = {
    CompressionMethod.GZIP: "#7B3294",
    CompressionMethod.ZSTD: "#008837",
}
_LEVEL_LINE_STYLES = ("-", "--", ":", "-.")

type _CompressionConfig = GzipConfig | ZstdConfig


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def _output_directory(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _compression_label(config: _CompressionConfig) -> str:
    return f"{config.method.value}:{config.level}"


def _compression_slug(config: _CompressionConfig) -> str:
    return f"{config.method.value}{config.level}"


def _compression_config(
    method: CompressionMethod,
    level: int,
) -> _CompressionConfig:
    try:
        if method is CompressionMethod.GZIP:
            return GzipConfig(level=level)
        return ZstdConfig(level=level)
    except ValidationError as exc:
        message = exc.errors(include_url=False)[0]["msg"]
        raise argparse.ArgumentTypeError(str(message)) from exc


def _compression_list(value: str) -> tuple[_CompressionConfig, ...]:
    entries = [entry.strip() for entry in value.split(",")]
    if not entries or any(not entry for entry in entries):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list such as gzip:6,zstd:3"
        )

    configs: list[_CompressionConfig] = []
    seen: set[tuple[CompressionMethod, int]] = set()
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"invalid compression {entry!r}; expected METHOD:LEVEL"
            )
        method_text, level_text = parts
        try:
            method = CompressionMethod(method_text)
        except ValueError as exc:
            choices = ", ".join(method.value for method in CompressionMethod)
            raise argparse.ArgumentTypeError(
                f"unknown compression method {method_text!r}; choose {choices}"
            ) from exc
        try:
            level = int(level_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"compression level must be an integer: {entry!r}"
            ) from exc
        identity = (method, level)
        if identity in seen:
            raise argparse.ArgumentTypeError(
                f"duplicate compression config: {entry!r}"
            )
        seen.add(identity)
        configs.append(_compression_config(method, level))
    return tuple(configs)


def _single_compression(value: str) -> _CompressionConfig:
    configs = _compression_list(value)
    if len(configs) != 1:
        raise argparse.ArgumentTypeError(
            "must be one compression config such as zstd:3"
        )
    return configs[0]


def _resolve_ordering_config(
    parser: argparse.ArgumentParser,
    configs: Sequence[_CompressionConfig],
    requested: _CompressionConfig | None,
) -> _CompressionConfig:
    configs_by_label = {
        _compression_label(config): config for config in configs
    }
    if requested is not None:
        requested_label = _compression_label(requested)
        if requested_label not in configs_by_label:
            parser.error(
                f"--order-by {requested_label!r} is not present in --comp"
            )
        return configs_by_label[requested_label]
    return configs_by_label.get(_DEFAULT_ORDERING, configs[0])


def _config_slug(configs: Sequence[_CompressionConfig]) -> str:
    return "-".join(_compression_slug(config) for config in configs)


def _default_output_directory(
    configs: Sequence[_CompressionConfig],
) -> Path:
    now = datetime.now().astimezone()
    return (
        _DATA_ROOT
        / now.strftime("%Y-%m-%d")
        / "figs"
        / f"{now:%H%M}-humaneval-compression-{_config_slug(configs)}"
    )


def _task_representations(task: HumanEvalTask) -> dict[str, str]:
    parsed = task.parsed
    code_without_comments = task.ground_truth_code_without_comments
    return {
        "code_without_comments": code_without_comments,
        "comments_and_docstrings": parsed.comments,
    }


def _measurement_table(
    tasks: Sequence[HumanEvalTask],
    configs: Sequence[_CompressionConfig],
) -> pl.DataFrame:
    rows: list[dict[str, str | int | float | bool]] = []
    for task in tasks:
        for representation, text in _task_representations(task).items():
            raw = text.encode("utf-8")
            raw_size = len(raw)
            if raw_size == 0:
                raise ValueError(
                    f"task {task.task_id!r} has an empty {representation}"
                )
            for config in configs:
                compressed_size = len(
                    compressed_bytes(
                        raw,
                        method=config.method,
                        level=config.level,
                    )
                )
                ratio = compressed_size / raw_size
                rows.append(
                    {
                        "task_id": task.task_id,
                        "representation": representation,
                        "method": config.method.value,
                        "level": config.level,
                        "configuration": _compression_label(config),
                        "raw_bytes": raw_size,
                        "compressed_bytes": compressed_size,
                        "bytes_saved": raw_size - compressed_size,
                        "compression_ratio": ratio,
                        "percent_reduction": (1.0 - ratio) * 100.0,
                        "is_smaller": compressed_size < raw_size,
                    }
                )
    return pl.DataFrame(rows)


def _linear_quantile(
    values: Sequence[int | float], probability: float
) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one value is required")
    position = (len(ordered) - 1) * probability
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def _summary_table(
    measurements: pl.DataFrame,
    configs: Sequence[_CompressionConfig],
) -> pl.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for representation in _REPRESENTATIONS:
        representation_rows = measurements.filter(
            pl.col("representation") == representation.key
        )
        for config in configs:
            label = _compression_label(config)
            selected = representation_rows.filter(
                pl.col("configuration") == label
            )
            compressed: list[int] = selected.get_column(
                "compressed_bytes"
            ).to_list()
            reductions: list[float] = selected.get_column(
                "percent_reduction"
            ).to_list()
            smaller_count = selected.get_column("is_smaller").sum()
            rows.append(
                {
                    "representation": representation.key,
                    "method": config.method.value,
                    "level": config.level,
                    "configuration": label,
                    "count": len(compressed),
                    "compressed_bytes_minimum": min(compressed),
                    "compressed_bytes_median": _linear_quantile(
                        compressed, 0.5
                    ),
                    "compressed_bytes_mean": sum(compressed) / len(compressed),
                    "compressed_bytes_p90": _linear_quantile(compressed, 0.9),
                    "compressed_bytes_p95": _linear_quantile(compressed, 0.95),
                    "compressed_bytes_maximum": max(compressed),
                    "percent_reduction_median": _linear_quantile(
                        reductions, 0.5
                    ),
                    "percent_reduction_mean": sum(reductions)
                    / len(reductions),
                    "percent_reduction_p10": _linear_quantile(reductions, 0.1),
                    "percent_reduction_p90": _linear_quantile(reductions, 0.9),
                    "smaller_count": int(smaller_count),
                    "smaller_rate": float(smaller_count) / len(compressed),
                }
            )
    return pl.DataFrame(rows)


def _style_violins(
    violins: Mapping[str, object],
    colors: Sequence[str],
) -> None:
    bodies = cast(Sequence[PolyCollection], violins["bodies"])
    for body, color in zip(bodies, colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)


def _jittered_violins(
    axis: Axes,
    values_by_series: Sequence[Sequence[int | float]],
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    seed: int,
    value_suffix: str,
) -> None:
    positions = list(range(1, len(values_by_series) + 1))
    violins = axis.violinplot(
        values_by_series,
        positions=positions,
        widths=0.72,
        showextrema=False,
    )
    _style_violins(violins, colors)
    jitter = random.Random(seed)
    tick_labels: list[str] = []
    for position, values, label, color in zip(
        positions,
        values_by_series,
        labels,
        colors,
        strict=True,
    ):
        x_values = [position + jitter.uniform(-0.1, 0.1) for _ in values]
        mean = sum(values) / len(values)
        median = _linear_quantile(values, 0.5)
        axis.scatter(
            x_values,
            values,
            s=13,
            alpha=0.16,
            color=color,
            edgecolors="none",
        )
        axis.scatter(
            [position],
            [mean],
            marker="D",
            s=30,
            color="#222222",
            zorder=3,
            label="Mean" if position == positions[0] else None,
        )
        axis.scatter(
            [position],
            [median],
            marker="_",
            s=250,
            linewidths=2,
            color="#222222",
            zorder=3,
            label="Median" if position == positions[0] else None,
        )
        tick_labels.append(f"{label}\nmedian={median:,.1f}{value_suffix}")
    axis.set_xticks(positions, tick_labels)


def _representation_measurements(
    measurements: pl.DataFrame,
    representation: _Representation,
    config: _CompressionConfig,
) -> pl.DataFrame:
    return measurements.filter(
        (pl.col("representation") == representation.key)
        & (pl.col("configuration") == _compression_label(config))
    )


def _compression_colors(
    configs: Sequence[_CompressionConfig],
) -> dict[str, str]:
    color_values = plt.colormaps["viridis"](
        [index / max(1, len(configs) - 1) for index in range(len(configs))]
    )
    return {
        _compression_label(config): matplotlib.colors.to_hex(color)
        for config, color in zip(configs, color_values, strict=True)
    }


def _configs_by_method(
    configs: Sequence[_CompressionConfig],
) -> dict[CompressionMethod, list[_CompressionConfig]]:
    grouped: dict[CompressionMethod, list[_CompressionConfig]] = {}
    for config in configs:
        grouped.setdefault(config.method, []).append(config)
    for method_configs in grouped.values():
        method_configs.sort(key=lambda config: config.level)
    return grouped


def _ecdf(values: Sequence[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one value is required")
    x_values = [ordered[0], *ordered]
    y_values = [
        0.0,
        *[index / len(ordered) for index in range(1, len(ordered) + 1)],
    ]
    return x_values, y_values


def _paired_ratio_differences(
    measurements: pl.DataFrame,
    candidate: _CompressionConfig,
    baseline: _CompressionConfig,
) -> tuple[float, ...]:
    code = _REPRESENTATIONS[0]
    candidate_rows = _representation_measurements(
        measurements, code, candidate
    ).select(
        "task_id",
        pl.col("compression_ratio").alias("candidate_ratio"),
    )
    baseline_rows = _representation_measurements(
        measurements, code, baseline
    ).select(
        "task_id",
        pl.col("compression_ratio").alias("baseline_ratio"),
    )
    paired = candidate_rows.join(
        baseline_rows,
        on="task_id",
        how="inner",
        validate="1:1",
    ).with_columns(
        (pl.col("candidate_ratio") - pl.col("baseline_ratio")).alias(
            "ratio_difference"
        )
    )
    if (
        paired.height != candidate_rows.height
        or paired.height != baseline_rows.height
    ):
        raise ValueError("compression configs do not cover the same tasks")
    return tuple(paired.get_column("ratio_difference").to_list())


def _paired_effect_curves(
    measurements: pl.DataFrame,
    configs: Sequence[_CompressionConfig],
) -> list[_EffectCurve]:
    grouped = _configs_by_method(configs)
    curves: list[_EffectCurve] = []
    for method in (CompressionMethod.GZIP, CompressionMethod.ZSTD):
        method_configs = grouped.get(method, [])
        if len(method_configs) < 2:
            continue
        baseline = method_configs[0]
        candidate = method_configs[-1]
        curves.append(
            _EffectCurve(
                label=(
                    f"{_compression_label(candidate)} − "
                    f"{_compression_label(baseline)}"
                ),
                values=_paired_ratio_differences(
                    measurements, candidate, baseline
                ),
                color=_ALGORITHM_COLORS[method],
                line_style="--",
            )
        )

    gzip_configs = grouped.get(CompressionMethod.GZIP, [])
    zstd_configs = grouped.get(CompressionMethod.ZSTD, [])
    if gzip_configs and zstd_configs:
        algorithm_pairs = (
            (zstd_configs[0], gzip_configs[0], "#2166AC"),
            (zstd_configs[-1], gzip_configs[-1], "#B2182B"),
        )
        seen: set[tuple[str, str]] = set()
        for candidate, baseline, color in algorithm_pairs:
            identity = (
                _compression_label(candidate),
                _compression_label(baseline),
            )
            if identity in seen:
                continue
            seen.add(identity)
            curves.append(
                _EffectCurve(
                    label=f"{identity[0]} − {identity[1]}",
                    values=_paired_ratio_differences(
                        measurements, candidate, baseline
                    ),
                    color=color,
                    line_style="-",
                )
            )
    return curves


def _plot_compression_ratio_ecdfs(
    measurements: pl.DataFrame,
    configs: Sequence[_CompressionConfig],
    *,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6.5),
        sharey=True,
    )
    grouped = _configs_by_method(configs)
    for method in (CompressionMethod.GZIP, CompressionMethod.ZSTD):
        method_configs = grouped.get(method, [])
        for index, config in enumerate(method_configs):
            selected = _representation_measurements(
                measurements,
                _REPRESENTATIONS[0],
                config,
            )
            ratios: list[float] = selected.get_column(
                "compression_ratio"
            ).to_list()
            x_values, y_values = _ecdf(ratios)
            axes[0].step(
                x_values,
                y_values,
                where="post",
                color=_ALGORITHM_COLORS[method],
                linestyle=_LEVEL_LINE_STYLES[index % len(_LEVEL_LINE_STYLES)],
                linewidth=2.2,
                label=_compression_label(config),
            )
    axes[0].axvline(
        1.0,
        linestyle=":",
        linewidth=1.4,
        color="#555555",
        label="No size change",
    )
    axes[0].set_title("Per-task compression-ratio distributions")
    axes[0].set_xlabel("Compression ratio (lower is better)")
    axes[0].set_ylabel("Fraction of tasks at or below x")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    effect_curves = _paired_effect_curves(measurements, configs)
    for curve in effect_curves:
        x_values, y_values = _ecdf(curve.values)
        axes[1].step(
            x_values,
            y_values,
            where="post",
            color=curve.color,
            linestyle=curve.line_style,
            linewidth=2.2,
            label=curve.label,
        )
    axes[1].axvline(
        0.0,
        linestyle=":",
        linewidth=1.4,
        color="#555555",
        label="No difference",
    )
    axes[1].set_title("Paired per-task effects")
    axes[1].set_xlabel(
        "Compression-ratio difference (first setting − second setting)\n"
        "Negative means the first setting is better"
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[0].set_ylim(0, 1.01)

    figure.suptitle(
        "HumanEval GT code ECDFs: compression algorithms versus levels"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _faceted_figure(count: int) -> tuple[Figure, list[Axes]]:
    column_count = min(2, count)
    row_count = math.ceil(count / column_count)
    figure, raw_axes = plt.subplots(
        row_count,
        column_count,
        figsize=(8 * column_count, 5.5 * row_count),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    return figure, cast(list[Axes], list(raw_axes.flat))


def _binned_medians(
    x_values: Sequence[int | float],
    y_values: Sequence[int | float],
    *,
    maximum_bins: int = 8,
) -> tuple[list[float], list[float]]:
    ordered = sorted(zip(x_values, y_values, strict=True))
    bin_count = min(maximum_bins, len(ordered))
    x_medians: list[float] = []
    y_medians: list[float] = []
    for bin_index in range(bin_count):
        start = bin_index * len(ordered) // bin_count
        end = (bin_index + 1) * len(ordered) // bin_count
        chunk = ordered[start:end]
        x_medians.append(_linear_quantile([pair[0] for pair in chunk], 0.5))
        y_medians.append(_linear_quantile([pair[1] for pair in chunk], 0.5))
    return x_medians, y_medians


def _plot_size_relationship_axis(
    axis: Axes,
    selected: pl.DataFrame,
    config: _CompressionConfig,
    metric: _ComparisonMetric,
    *,
    color: str,
) -> None:
    raw_bytes: list[int] = selected.get_column("raw_bytes").to_list()
    values: list[int | float] = selected.get_column(metric.key).to_list()
    median_x, median_y = _binned_medians(raw_bytes, values)
    axis.scatter(
        raw_bytes,
        values,
        s=22,
        alpha=0.28,
        color=color,
        edgecolors="none",
        label="Task",
    )
    axis.plot(
        median_x,
        median_y,
        color="#222222",
        linewidth=1.6,
        marker="o",
        markersize=4,
        label="Equal-count bin median",
    )
    axis.axhline(
        metric.reference_value,
        linestyle="--",
        linewidth=1.2,
        color="#666666",
        label=metric.reference_label,
    )
    axis.set_xscale("log")
    axis.set_title(_compression_label(config))
    axis.set_xlabel("Initial normalized GT code size (bytes, log scale)")
    axis.set_ylabel(metric.label)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def _save_size_relationship_plots(
    measurements: pl.DataFrame,
    configs: Sequence[_CompressionConfig],
    *,
    output_dir: Path,
    config_slug: str,
) -> list[tuple[str, Path]]:
    code = _REPRESENTATIONS[0]
    colors = _compression_colors(configs)
    written: list[tuple[str, Path]] = []
    selected_by_label = {
        _compression_label(config): _representation_measurements(
            measurements, code, config
        )
        for config in configs
    }
    for metric in _COMPARISON_METRICS:
        for config in configs:
            label = _compression_label(config)
            path = output_dir / (
                "humaneval_code_raw_bytes_vs_"
                f"{metric.filename_slug}_{_compression_slug(config)}.png"
            )
            figure, axis = plt.subplots(figsize=(8, 5.5))
            _plot_size_relationship_axis(
                axis,
                selected_by_label[label],
                config,
                metric,
                color=colors[label],
            )
            figure.suptitle(
                f"HumanEval GT code: initial size vs {metric.label.lower()}"
            )
            figure.tight_layout()
            figure.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(figure)
            written.append(
                (
                    f"{label} initial-size/{metric.filename_slug} plot",
                    path,
                )
            )

        faceted_path = output_dir / (
            "humaneval_code_raw_bytes_vs_"
            f"{metric.filename_slug}_faceted_{config_slug}.png"
        )
        figure, axes = _faceted_figure(len(configs))
        for axis, config in zip(axes, configs, strict=False):
            label = _compression_label(config)
            _plot_size_relationship_axis(
                axis,
                selected_by_label[label],
                config,
                metric,
                color=colors[label],
            )
        for axis in axes[len(configs) :]:
            axis.set_visible(False)
        figure.suptitle(
            f"HumanEval GT code: initial size vs {metric.label.lower()}"
        )
        figure.tight_layout()
        figure.savefig(faceted_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append(
            (f"faceted initial-size/{metric.filename_slug} plot", faceted_path)
        )
    return written


def _task_order_table(
    measurements: pl.DataFrame,
    order_by: _CompressionConfig,
) -> pl.DataFrame:
    selected = _representation_measurements(
        measurements,
        _REPRESENTATIONS[0],
        order_by,
    ).sort(
        ["compression_ratio", "task_id"],
        descending=[True, False],
    )
    return selected.with_row_index("task_rank", offset=1).select(
        "task_rank",
        "task_id",
        "raw_bytes",
        pl.lit(_compression_label(order_by)).alias("order_by_config"),
        pl.col("compression_ratio").alias("order_by_compression_ratio"),
        pl.col("bytes_saved").alias("order_by_bytes_saved"),
    )


def _plot_ordered_axis(
    axis: Axes,
    measurements: pl.DataFrame,
    task_order: pl.DataFrame,
    config: _CompressionConfig,
    metric: _ComparisonMetric,
    *,
    color: str,
) -> None:
    selected = (
        _representation_measurements(
            measurements,
            _REPRESENTATIONS[0],
            config,
        )
        .join(
            task_order.select("task_id", "task_rank"),
            on="task_id",
            how="inner",
        )
        .sort("task_rank")
    )
    ranks: list[int] = selected.get_column("task_rank").to_list()
    values: list[int | float] = selected.get_column(metric.key).to_list()
    axis.plot(ranks, values, color=color, linewidth=1.0, alpha=0.55)
    axis.scatter(
        ranks,
        values,
        s=17,
        alpha=0.34,
        color=color,
        edgecolors="none",
    )
    axis.axhline(
        metric.reference_value,
        linestyle="--",
        linewidth=1.2,
        color="#666666",
        label=metric.reference_label,
    )
    axis.set_title(_compression_label(config))
    axis.set_ylabel(metric.label)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)


def _save_ordered_comparison_plots(
    measurements: pl.DataFrame,
    task_order: pl.DataFrame,
    configs: Sequence[_CompressionConfig],
    order_by: _CompressionConfig,
    *,
    output_dir: Path,
    config_slug: str,
) -> list[tuple[str, Path]]:
    order_label = _compression_label(order_by)
    order_slug = _compression_slug(order_by)
    colors = _compression_colors(configs)
    written: list[tuple[str, Path]] = []
    for metric in _COMPARISON_METRICS:
        path = output_dir / (
            f"humaneval_code_ordered_by_{order_slug}_"
            f"{metric.filename_slug}_{config_slug}.png"
        )
        figure, axes = _faceted_figure(len(configs))
        for axis, config in zip(axes, configs, strict=False):
            label = _compression_label(config)
            _plot_ordered_axis(
                axis,
                measurements,
                task_order,
                config,
                metric,
                color=colors[label],
            )
        for axis in axes[len(configs) :]:
            axis.set_visible(False)
        figure.supxlabel(
            f"Task rank under {order_label} "
            "(least compressible to most compressible)"
        )
        figure.suptitle(f"HumanEval GT code: aligned {metric.label.lower()}")
        figure.tight_layout()
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append((f"ordered {metric.filename_slug} plot", path))
    return written


def _plot_representation(
    measurements: pl.DataFrame,
    representation: _Representation,
    configs: Sequence[_CompressionConfig],
    *,
    path: Path,
) -> None:
    selected_by_config = [
        _representation_measurements(measurements, representation, config)
        for config in configs
    ]
    raw: list[int] = selected_by_config[0].get_column("raw_bytes").to_list()
    compressed: list[list[int]] = [
        selected.get_column("compressed_bytes").to_list()
        for selected in selected_by_config
    ]
    reductions: list[list[float]] = [
        selected.get_column("percent_reduction").to_list()
        for selected in selected_by_config
    ]
    labels = [_compression_label(config) for config in configs]
    compression_colors = plt.colormaps["viridis"](
        [index / max(1, len(configs) - 1) for index in range(len(configs))]
    )
    colors = [matplotlib.colors.to_hex(color) for color in compression_colors]

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    _jittered_violins(
        axes[0],
        [raw, *compressed],
        ["raw", *labels],
        ["#777777", *colors],
        seed=0,
        value_suffix=" B",
    )
    axes[0].set_title("Raw and compressed sizes")
    axes[0].set_ylabel("Bytes per task")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    _jittered_violins(
        axes[1],
        reductions,
        labels,
        colors,
        seed=1,
        value_suffix="%",
    )
    axes[1].axhline(
        0,
        linestyle="--",
        linewidth=1.2,
        color="#555555",
        label="No size change",
    )
    axes[1].set_title("Per-task reduction from raw size")
    axes[1].set_ylabel("Percent reduction")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    figure.suptitle(f"HumanEval compression: {representation.label}")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _summary_lines(
    summary: pl.DataFrame,
    configs: Sequence[_CompressionConfig],
) -> list[str]:
    lines = [
        "Compression configs: "
        + ", ".join(_compression_label(config) for config in configs)
    ]
    for representation in _REPRESENTATIONS:
        lines.append(f"{representation.label}:")
        representation_rows = summary.filter(
            pl.col("representation") == representation.key
        )
        for config in configs:
            row = representation_rows.filter(
                pl.col("configuration") == _compression_label(config)
            ).row(0, named=True)
            lines.append(
                f"  {_compression_label(config)}: "
                f"median compressed="
                f"{float(row['compressed_bytes_median']):,.1f} B, "
                f"median reduction="
                f"{float(row['percent_reduction_median']):,.1f}%, "
                f"smaller={int(row['smaller_count']):,}/"
                f"{int(row['count']):,} "
                f"({float(row['smaller_rate']):.1%})"
            )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure and plot per-task compression of normalized HumanEval "
            "code and comments."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=_existing_file,
        default=_DEFAULT_SNAPSHOT,
        help=f"HumanEval snapshot path (default: {_DEFAULT_SNAPSHOT})",
    )
    parser.add_argument(
        "--comp",
        type=_compression_list,
        default=_compression_list(_DEFAULT_COMPRESSIONS),
        help=(
            "comma-separated METHOD:LEVEL list "
            f"(default: {_DEFAULT_COMPRESSIONS})"
        ),
    )
    parser.add_argument(
        "--order-by",
        type=_single_compression,
        help=(
            "configured method used to rank tasks from least to most "
            f"compressible (default: {_DEFAULT_ORDERING} when present, "
            "otherwise the first --comp entry)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=_output_directory,
        help=(
            "output directory; defaults to a dated run directory under "
            f"{_DATA_ROOT}"
        ),
    )
    arguments = parser.parse_args()
    configs: tuple[_CompressionConfig, ...] = arguments.comp
    order_by = _resolve_ordering_config(parser, configs, arguments.order_by)
    output_dir = (
        arguments.output_dir
        if arguments.output_dir is not None
        else _default_output_directory(configs)
    )

    tasks = parse_humaneval_dataset(
        load_humaneval_rows(snapshot_path=arguments.snapshot)
    )
    if not tasks:
        parser.error("HumanEval snapshot contains no tasks")

    measurements = _measurement_table(tasks, configs)
    summary = _summary_table(measurements, configs)
    task_order = _task_order_table(measurements, order_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _config_slug(configs)
    measurements_path = (
        output_dir / f"humaneval_compression_measurements_bytes_{slug}.csv"
    )
    summary_path = (
        output_dir / f"humaneval_compression_summary_bytes_{slug}.csv"
    )
    task_order_path = output_dir / (
        f"humaneval_code_task_order_by_{_compression_slug(order_by)}_{slug}.csv"
    )
    code_plot_path = (
        output_dir / f"humaneval_code_compression_bytes_{slug}.png"
    )
    comments_plot_path = (
        output_dir
        / f"humaneval_comments_docstrings_compression_bytes_{slug}.png"
    )
    ecdf_plot_path = (
        output_dir
        / f"humaneval_code_compression_ratio_ecdf_comparison_{slug}.png"
    )
    log_path = output_dir / f"humaneval_compression_analysis_bytes_{slug}.log"

    measurements.write_csv(measurements_path)
    summary.write_csv(summary_path)
    task_order.write_csv(task_order_path)
    _plot_representation(
        measurements,
        _REPRESENTATIONS[0],
        configs,
        path=code_plot_path,
    )
    _plot_representation(
        measurements,
        _REPRESENTATIONS[1],
        configs,
        path=comments_plot_path,
    )
    comparison_plots = [
        *_save_size_relationship_plots(
            measurements,
            configs,
            output_dir=output_dir,
            config_slug=slug,
        ),
        *_save_ordered_comparison_plots(
            measurements,
            task_order,
            configs,
            order_by,
            output_dir=output_dir,
            config_slug=slug,
        ),
    ]
    _plot_compression_ratio_ecdfs(
        measurements,
        configs,
        path=ecdf_plot_path,
    )
    comparison_plots.append(
        ("algorithm/level compression-ratio ECDF comparison", ecdf_plot_path)
    )

    report = [
        f"Loaded {len(tasks):,} HumanEval tasks from {arguments.snapshot}",
        *_summary_lines(summary, configs),
        "Task ordering: "
        f"{_compression_label(order_by)} ratio descending "
        "(least to most compressible)",
        f"Wrote task measurements: {measurements_path}",
        f"Wrote aggregate summary: {summary_path}",
        f"Wrote task ordering: {task_order_path}",
        f"Wrote code compression plot: {code_plot_path}",
        f"Wrote comments/docstrings compression plot: {comments_plot_path}",
        *[
            f"Wrote {description}: {path}"
            for description, path in comparison_plots
        ],
        f"Wrote output log: {log_path}",
    ]
    output = "\n".join(report) + "\n"
    log_path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
