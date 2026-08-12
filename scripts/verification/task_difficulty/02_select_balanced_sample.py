#!/usr/bin/env python3

"""Select one deterministic generation per task and populated setting group."""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from workflow_settings import (
    ELIGIBLE_CORPUS,
    SAMPLE_TASKS_PER_GROUP,
    SAMPLE_TASKS_PER_GROUP_ENV,
    SAMPLING_COVERAGE,
    SAMPLING_LOG,
    SAMPLING_SEED,
    SELECTED_SAMPLE,
    parse_sample_tasks_per_group,
    prepare_run_directory,
    sample_tasks_per_group,
)

_SETTING_COLUMNS = (
    "generation_mode",
    "budget_mode",
    "model_key",
)
_GROUP_COLUMNS = ("task_id", *_SETTING_COLUMNS)
_REQUIRED_COLUMNS = frozenset(
    {*_GROUP_COLUMNS, "sample_id", "code_candidates", "candidate_count"}
)


def _configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("task_difficulty.sample")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(path, mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _sampling_key(sample_id: str) -> str:
    payload = f"{SAMPLING_SEED}\0{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _task_ordering_key(
    generation_mode: str,
    budget_mode: str,
    model_key: str,
    task_id: str,
) -> str:
    payload = "\0".join(
        (
            str(SAMPLING_SEED),
            generation_mode,
            budget_mode,
            model_key,
            task_id,
        )
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def select_balanced_sample(
    eligible: pl.DataFrame,
    *,
    tasks_per_group: int | None = SAMPLE_TASKS_PER_GROUP,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Select one generation per (task, setting-group) cell.

    Each populated `(generation_mode, budget_mode, model_key)` group keeps a
    seeded deterministic subset of `tasks_per_group` tasks (`None` keeps every
    task), then contributes one deterministic generation per retained task.
    """

    missing = _REQUIRED_COLUMNS.difference(eligible.columns)
    if missing:
        raise ValueError(
            "eligible corpus is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if eligible.get_column("sample_id").null_count():
        raise ValueError("eligible corpus contains null sample IDs")
    if tasks_per_group is not None and tasks_per_group < 1:
        raise ValueError("tasks_per_group must be positive or None")

    keyed = eligible.with_columns(
        pl.col("sample_id")
        .map_elements(_sampling_key, return_dtype=pl.String)
        .alias("sampling_key")
    )
    cells = (
        keyed.sort([*_GROUP_COLUMNS, "sampling_key"])
        .unique(subset=list(_GROUP_COLUMNS), keep="first", maintain_order=True)
        .sort(list(_GROUP_COLUMNS))
    )
    if cells.is_empty():
        raise ValueError(
            "eligible corpus has no populated "
            "(generation_mode, budget_mode, model_key) group; coverage "
            f"matrix:\n{_coverage_matrix(eligible)}"
        )

    if tasks_per_group is None:
        selected_cells = cells
    else:
        selected_cells = (
            cells.with_columns(
                pl.struct([*_SETTING_COLUMNS, "task_id"])
                .map_elements(
                    lambda values: _task_ordering_key(
                        values["generation_mode"],
                        values["budget_mode"],
                        values["model_key"],
                        values["task_id"],
                    ),
                    return_dtype=pl.String,
                )
                .alias("task_ordering_key")
            )
            .sort([*_SETTING_COLUMNS, "task_ordering_key"])
            .group_by(list(_SETTING_COLUMNS), maintain_order=True)
            .head(tasks_per_group)
            .drop("task_ordering_key")
            .sort(list(_GROUP_COLUMNS))
        )

    selected = selected_cells.with_columns(
        pl.lit(SAMPLING_SEED).alias("sampling_seed"),
        pl.lit(tasks_per_group, dtype=pl.Int64).alias("tasks_per_group"),
    )
    coverage = (
        selected.group_by(list(_SETTING_COLUMNS))
        .agg(
            pl.len().alias("selected_tasks"),
            pl.col("candidate_count").sum().alias("candidate_count"),
        )
        .join(
            cells.group_by(list(_SETTING_COLUMNS)).agg(
                pl.len().alias("eligible_tasks")
            ),
            on=list(_SETTING_COLUMNS),
            how="left",
        )
        .sort(list(_SETTING_COLUMNS))
    )
    return selected, coverage


def _coverage_matrix(eligible: pl.DataFrame) -> pl.DataFrame:
    present = [
        column for column in _SETTING_COLUMNS if column in eligible.columns
    ]
    if not present:
        return eligible.head(0)
    return eligible.group_by(present).len().sort(present)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-per-group",
        type=parse_sample_tasks_per_group,
        default=sample_tasks_per_group(),
        help=(
            "tasks retained per populated (generation_mode, budget_mode, "
            "model_key) group; pass 0 or 'all' to keep every task "
            f"(default: {SAMPLE_TASKS_PER_GROUP}, override with "
            f"{SAMPLE_TASKS_PER_GROUP_ENV})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    prepare_run_directory()
    logger = _configure_logging(SAMPLING_LOG)
    logger.info("Loading %s", ELIGIBLE_CORPUS)
    eligible = pl.read_parquet(ELIGIBLE_CORPUS)
    selected, coverage = select_balanced_sample(
        eligible,
        tasks_per_group=arguments.tasks_per_group,
    )
    selected.write_parquet(SELECTED_SAMPLE)
    coverage.write_parquet(SAMPLING_COVERAGE)
    logger.info(
        "Selected %d generations containing %d candidates across %d groups "
        "with tasks_per_group=%s",
        selected.height,
        selected.get_column("candidate_count").sum(),
        coverage.height,
        "all"
        if arguments.tasks_per_group is None
        else (arguments.tasks_per_group),
    )
    logger.info("Per-group coverage:\n%s", coverage)
    logger.info("Wrote sample to %s", SELECTED_SAMPLE)
    logger.info("Wrote coverage to %s", SAMPLING_COVERAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
