#!/usr/bin/env python3

"""Summarize and plot normalized HumanEval ground-truth source lengths."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import polars as pl

from dr_code.core.source.python_analysis import (
    SourceTextSite,
    extract_docstrings,
    extract_hash_comments,
)
from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.sampling import load_humaneval_rows

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_ROOT = Path(__file__).parents[1]
_DEFAULT_SNAPSHOT = _ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
_DEFAULT_OUTPUT_DIR = _ROOT / ".runs" / "humaneval-lengths"
_UNITS = ("characters", "bytes")


@dataclass(frozen=True, slots=True)
class _Representation:
    key: str
    label: str


_REPRESENTATIONS = (
    _Representation("ground_truth", "Full ground-truth source"),
    _Representation(
        "code_without_comments",
        "Normalized code without comments or docstrings",
    ),
    _Representation("docstrings", "Normalized docstring content"),
    _Representation("hash_comments", "Normalized hash-comment content"),
    _Representation(
        "comments_and_docstrings",
        "Normalized comments and docstrings",
    ),
)
_PLOTTED_REPRESENTATIONS = (
    _REPRESENTATIONS[1],
    _REPRESENTATIONS[4],
)


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def _output_directory(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _site_text(sites: Sequence[SourceTextSite]) -> str:
    ordered = sorted(
        sites,
        key=lambda site: (site.location.lineno, site.location.col_offset),
    )
    return "\n".join(site.text for site in ordered)


def _lengths(text: str) -> tuple[int, int]:
    return len(text), len(text.encode("utf-8"))


def _task_row(task: HumanEvalTask) -> dict[str, str | int]:
    parsed = task.parsed
    code_without_comments = task.ground_truth_code_without_comments
    if parsed is None or code_without_comments is None:
        raise ValueError(f"task {task.task_id!r} has no parsed source")

    tree = ast.parse(task.ground_truth_code)
    representations = {
        "ground_truth": task.ground_truth_code,
        "code_without_comments": code_without_comments,
        "docstrings": _site_text(extract_docstrings(tree)),
        "hash_comments": _site_text(
            extract_hash_comments(task.ground_truth_code)
        ),
        "comments_and_docstrings": parsed.comments,
    }
    row: dict[str, str | int] = {"task_id": task.task_id}
    for key, text in representations.items():
        characters, byte_count = _lengths(text)
        row[f"{key}_characters"] = characters
        row[f"{key}_bytes"] = byte_count
    return row


def _measurement_table(tasks: Sequence[HumanEvalTask]) -> pl.DataFrame:
    return pl.DataFrame([_task_row(task) for task in tasks])


def _linear_quantile(values: Sequence[int], probability: float) -> float:
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


def _summary_table(measurements: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for representation in _REPRESENTATIONS:
        for unit in _UNITS:
            values: list[int] = measurements.get_column(
                f"{representation.key}_{unit}"
            ).to_list()
            rows.append(
                {
                    "representation": representation.key,
                    "unit": unit,
                    "count": len(values),
                    "minimum": min(values),
                    "median": _linear_quantile(values, 0.5),
                    "mean": sum(values) / len(values),
                    "p90": _linear_quantile(values, 0.9),
                    "p95": _linear_quantile(values, 0.95),
                    "maximum": max(values),
                }
            )
    return pl.DataFrame(rows)


def _plot_histograms(
    measurements: pl.DataFrame,
    *,
    unit: str,
    bins: int,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ("#31688e", "#35b779")
    for axis, representation, color in zip(
        axes,
        _PLOTTED_REPRESENTATIONS,
        colors,
        strict=True,
    ):
        values = measurements.get_column(
            f"{representation.key}_{unit}"
        ).to_list()
        axis.hist(values, bins=bins, color=color, edgecolor="white")
        axis.set_title(representation.label)
        axis.set_xlabel(unit.capitalize())
        axis.set_ylabel("Tasks")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("HumanEval normalized ground-truth length distributions")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_ecdf(
    measurements: pl.DataFrame,
    *,
    unit: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for representation in _PLOTTED_REPRESENTATIONS:
        values = sorted(
            measurements.get_column(f"{representation.key}_{unit}").to_list()
        )
        cumulative = [
            index / len(values) for index in range(1, len(values) + 1)
        ]
        axis.step(
            values,
            cumulative,
            where="post",
            label=representation.label,
        )
    axis.set_title("HumanEval normalized ground-truth length ECDF")
    axis.set_xlabel(unit.capitalize())
    axis.set_ylabel("Fraction of tasks")
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_comments_vs_code(
    measurements: pl.DataFrame,
    *,
    unit: str,
    path: Path,
) -> None:
    code = measurements.get_column(f"code_without_comments_{unit}").to_list()
    comments = measurements.get_column(
        f"comments_and_docstrings_{unit}"
    ).to_list()
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(code, comments, alpha=0.7, color="#3b528b")
    axis.set_title("HumanEval comments versus normalized code length")
    axis.set_xlabel(f"Code without comments or docstrings ({unit})")
    axis.set_ylabel(f"Comments and docstrings ({unit})")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _print_summary(summary: pl.DataFrame, *, unit: str) -> None:
    print(f"Summary ({unit}):")
    selected = summary.filter(pl.col("unit") == unit)
    labels = {item.key: item.label for item in _REPRESENTATIONS}
    for row in selected.iter_rows(named=True):
        print(
            f"  {labels[str(row['representation'])]}: "
            f"min={float(row['minimum']):,.0f}, "
            f"median={float(row['median']):,.1f}, "
            f"mean={float(row['mean']):,.1f}, "
            f"p90={float(row['p90']):,.1f}, "
            f"p95={float(row['p95']):,.1f}, "
            f"max={float(row['maximum']):,.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure and plot normalized HumanEval ground-truth source, "
            "comment, and docstring lengths."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=_existing_file,
        default=_DEFAULT_SNAPSHOT,
        help=f"HumanEval snapshot path (default: {_DEFAULT_SNAPSHOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=_output_directory,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--unit",
        choices=_UNITS,
        default="characters",
        help="unit used for plots and printed summary (default: characters)",
    )
    parser.add_argument(
        "--bins",
        type=_positive_int,
        default=20,
        help="number of bins in each histogram (default: 20)",
    )
    arguments = parser.parse_args()

    tasks = parse_humaneval_dataset(
        load_humaneval_rows(snapshot_path=arguments.snapshot)
    )
    if not tasks:
        parser.error("HumanEval snapshot contains no tasks")

    measurements = _measurement_table(tasks)
    summary = _summary_table(measurements)
    output_dir: Path = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements_path = output_dir / "humaneval_task_lengths.csv"
    summary_path = output_dir / "humaneval_length_summary.csv"
    histograms_path = (
        output_dir / f"humaneval_length_histograms_{arguments.unit}.png"
    )
    ecdf_path = output_dir / f"humaneval_length_ecdf_{arguments.unit}.png"
    scatter_path = (
        output_dir / f"humaneval_comments_vs_code_{arguments.unit}.png"
    )

    measurements.write_csv(measurements_path)
    summary.write_csv(summary_path)
    _plot_histograms(
        measurements,
        unit=arguments.unit,
        bins=arguments.bins,
        path=histograms_path,
    )
    _plot_ecdf(measurements, unit=arguments.unit, path=ecdf_path)
    _plot_comments_vs_code(
        measurements,
        unit=arguments.unit,
        path=scatter_path,
    )

    print(f"Loaded {len(tasks):,} HumanEval tasks from {arguments.snapshot}")
    _print_summary(summary, unit=arguments.unit)
    print(f"Wrote task measurements: {measurements_path}")
    print(f"Wrote aggregate summary: {summary_path}")
    print(f"Wrote histogram plot: {histograms_path}")
    print(f"Wrote ECDF plot: {ecdf_path}")
    print(f"Wrote scatter plot: {scatter_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
