#!/usr/bin/env python3

"""Measure compression of normalized HumanEval code and comments."""

from __future__ import annotations

import argparse
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

_ROOT = Path(__file__).parents[1]
_DEFAULT_SNAPSHOT = _ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
_DATA_ROOT = Path.home() / "drotherm" / "data" / ".codex" / "dr-code"
_DEFAULT_COMPRESSIONS = "gzip:6,gzip:9,zstd:3,zstd:9"


@dataclass(frozen=True, slots=True)
class _Representation:
    key: str
    label: str


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
                f"duplicate compression configuration: {entry!r}"
            )
        seen.add(identity)
        configs.append(_compression_config(method, level))
    return tuple(configs)


def _configuration_slug(configs: Sequence[_CompressionConfig]) -> str:
    return "-".join(_compression_slug(config) for config in configs)


def _default_output_directory(
    configs: Sequence[_CompressionConfig],
) -> Path:
    now = datetime.now().astimezone()
    return (
        _DATA_ROOT
        / now.strftime("%Y-%m-%d")
        / "figs"
        / f"{now:%H%M}-humaneval-compression-{_configuration_slug(configs)}"
    )


def _task_representations(task: HumanEvalTask) -> dict[str, str]:
    parsed = task.parsed
    code_without_comments = task.ground_truth_code_without_comments
    if parsed is None or code_without_comments is None:
        raise ValueError(f"task {task.task_id!r} has no parsed source")
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
        "Compression configurations: "
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
        "--output-dir",
        type=_output_directory,
        help=(
            "output directory; defaults to a dated run directory under "
            f"{_DATA_ROOT}"
        ),
    )
    arguments = parser.parse_args()
    configs: tuple[_CompressionConfig, ...] = arguments.comp
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
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _configuration_slug(configs)
    measurements_path = (
        output_dir / f"humaneval_compression_measurements_bytes_{slug}.csv"
    )
    summary_path = (
        output_dir / f"humaneval_compression_summary_bytes_{slug}.csv"
    )
    code_plot_path = (
        output_dir / f"humaneval_code_compression_bytes_{slug}.png"
    )
    comments_plot_path = (
        output_dir
        / f"humaneval_comments_docstrings_compression_bytes_{slug}.png"
    )
    log_path = output_dir / f"humaneval_compression_analysis_bytes_{slug}.log"

    measurements.write_csv(measurements_path)
    summary.write_csv(summary_path)
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

    report = [
        f"Loaded {len(tasks):,} HumanEval tasks from {arguments.snapshot}",
        *_summary_lines(summary, configs),
        f"Wrote task measurements: {measurements_path}",
        f"Wrote aggregate summary: {summary_path}",
        f"Wrote code compression plot: {code_plot_path}",
        f"Wrote comments/docstrings compression plot: {comments_plot_path}",
        f"Wrote output log: {log_path}",
    ]
    output = "\n".join(report) + "\n"
    log_path.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
