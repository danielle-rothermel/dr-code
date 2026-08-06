#!/usr/bin/env python3

"""Select one eligible generation per task, setting, and fixed model."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import polars as pl

from workflow_settings import (
    ELIGIBLE_CORPUS,
    MODEL_ROSTER,
    SAMPLING_COVERAGE,
    SAMPLING_LOG,
    SAMPLING_SEED,
    SELECTED_SAMPLE,
    SETTINGS,
    prepare_run_directory,
)

_GROUP_COLUMNS = (
    "task_id",
    "generation_mode",
    "budget_mode",
    "model_key",
)
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


def select_balanced_sample(
    eligible: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    missing = _REQUIRED_COLUMNS.difference(eligible.columns)
    if missing:
        raise ValueError(
            "eligible corpus is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if eligible.get_column("sample_id").null_count():
        raise ValueError("eligible corpus contains null sample IDs")

    roster = eligible.filter(
        pl.col("model_key").is_in(MODEL_ROSTER)
        & pl.struct(["generation_mode", "budget_mode"]).is_in(
            [
                {
                    "generation_mode": generation_mode,
                    "budget_mode": budget_mode,
                }
                for generation_mode, budget_mode in SETTINGS
            ]
        )
    ).with_columns(
        pl.col("sample_id")
        .map_elements(_sampling_key, return_dtype=pl.String)
        .alias("sampling_key")
    )
    selected = (
        roster.sort([*_GROUP_COLUMNS, "sampling_key"])
        .unique(subset=list(_GROUP_COLUMNS), keep="first", maintain_order=True)
        .with_columns(pl.lit(SAMPLING_SEED).alias("sampling_seed"))
        .sort(list(_GROUP_COLUMNS))
    )

    task_ids = eligible.get_column("task_id").unique().sort().to_list()
    expected = pl.DataFrame(
        [
            {
                "task_id": task_id,
                "generation_mode": generation_mode,
                "budget_mode": budget_mode,
                "model_key": model,
            }
            for task_id in task_ids
            for generation_mode, budget_mode in SETTINGS
            for model in MODEL_ROSTER
        ]
    )
    coverage = expected.join(
        selected.select(_GROUP_COLUMNS).with_columns(
            pl.lit(True).alias("selected")
        ),
        on=list(_GROUP_COLUMNS),
        how="left",
    ).with_columns(pl.col("selected").fill_null(False))
    return selected, coverage


def main() -> int:
    prepare_run_directory()
    logger = _configure_logging(SAMPLING_LOG)
    logger.info("Loading %s", ELIGIBLE_CORPUS)
    eligible = pl.read_parquet(ELIGIBLE_CORPUS)
    selected, coverage = select_balanced_sample(eligible)
    selected.write_parquet(SELECTED_SAMPLE)
    coverage.write_parquet(SAMPLING_COVERAGE)
    missing_cells = coverage.filter(~pl.col("selected")).height
    logger.info(
        "Selected %d generations containing %d candidates",
        selected.height,
        selected.get_column("candidate_count").sum(),
    )
    logger.info("Unavailable task-setting-model cells: %d", missing_cells)
    logger.info("Wrote sample to %s", SELECTED_SAMPLE)
    logger.info("Wrote coverage to %s", SAMPLING_COVERAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
