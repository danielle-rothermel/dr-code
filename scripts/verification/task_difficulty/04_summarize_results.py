#!/usr/bin/env python3

"""Summarize completed candidate evaluations at generation and task level."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from workflow_settings import (
    PREPROCESSING_SUMMARY,
    SELECTED_SAMPLE,
    evaluation_paths,
    parse_evaluation_args,
    prepare_run_directory,
)

_GENERATION_KEYS = (
    "sample_id",
    "task_id",
    "generation_mode",
    "budget_mode",
    "model_key",
)


def _configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("task_difficulty.summarize")
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


def _wilson_interval(passed: int, total: int) -> tuple[float, float]:
    if total < 1:
        return (math.nan, math.nan)
    z = 1.959963984540054
    rate = passed / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return (center - margin, center + margin)


def summarize_results(
    selected: pl.DataFrame,
    candidate_results: pl.DataFrame,
    preprocessing_summary: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    observed = candidate_results.group_by(list(_GENERATION_KEYS)).agg(
        pl.len().alias("evaluated_candidate_count"),
        (pl.col("metric_status") == "measured")
        .sum()
        .alias("measured_candidate_count"),
        pl.col("candidate_passed")
        .fill_null(False)
        .any()
        .alias("any_candidate_passed"),
    )
    generations = (
        selected.select([*_GENERATION_KEYS, "candidate_count"])
        .join(observed, on=list(_GENERATION_KEYS), how="left")
        .with_columns(
            pl.col("evaluated_candidate_count").fill_null(0),
            pl.col("measured_candidate_count").fill_null(0),
            pl.col("any_candidate_passed").fill_null(False),
        )
        .with_columns(
            (
                (
                    pl.col("evaluated_candidate_count")
                    == pl.col("candidate_count")
                )
                & (
                    pl.col("measured_candidate_count")
                    == pl.col("candidate_count")
                )
            ).alias("evaluation_complete")
        )
        .with_columns(
            pl.when(pl.col("evaluation_complete"))
            .then(pl.col("any_candidate_passed"))
            .otherwise(None)
            .alias("generation_passed")
        )
        .sort(list(_GENERATION_KEYS))
    )
    complete = generations.filter(pl.col("evaluation_complete"))
    if complete.is_empty():
        raise ValueError("no generation has a complete metric evaluation")
    task_settings = (
        complete.group_by(["task_id", "generation_mode", "budget_mode"])
        .agg(
            pl.len().alias("evaluated_generations"),
            pl.col("generation_passed").sum().alias("passed_generations"),
        )
        .with_columns(
            (
                pl.col("passed_generations") / pl.col("evaluated_generations")
            ).alias("test_success_rate")
        )
        .sort(["task_id", "generation_mode", "budget_mode"])
    )

    task_core = (
        complete.group_by("task_id")
        .agg(
            pl.len().alias("evaluated_generations"),
            pl.col("generation_passed").sum().alias("passed_generations"),
        )
        .sort("task_id")
    )
    task_records: list[dict[str, object]] = []
    for row in task_core.iter_rows(named=True):
        total = int(row["evaluated_generations"])
        passed = int(row["passed_generations"])
        lower, upper = _wilson_interval(passed, total)
        if passed == 0:
            observed_extreme = "all_failed"
        elif passed == total:
            observed_extreme = "all_passed"
        else:
            observed_extreme = "mixed"
        task_records.append(
            {
                **row,
                "test_success_rate": passed / total,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "observed_extreme": observed_extreme,
            }
        )
    tasks = pl.DataFrame(task_records, infer_schema_length=None)
    preprocessing_by_task = (
        preprocessing_summary.group_by("task_id")
        .agg(
            pl.col("nonblank_rows").sum().alias("nonblank_rows"),
            pl.col("eligible_rows").sum().alias("eligible_rows"),
        )
        .with_columns(
            (pl.col("eligible_rows") / pl.col("nonblank_rows")).alias(
                "preprocessing_success_rate"
            )
        )
    )
    tasks = tasks.join(preprocessing_by_task, on="task_id", how="left").sort(
        ["test_success_rate", "task_id"]
    )
    return generations, task_settings, tasks


def main(argv: Sequence[str] | None = None) -> int:
    settings = parse_evaluation_args(__doc__, argv)
    paths = evaluation_paths(settings)
    prepare_run_directory()
    paths.root.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(paths.summary_log)
    if paths.candidate_results.is_file():
        candidate_results = pl.read_parquet(paths.candidate_results)
    else:
        raise SystemExit(
            f"candidate results not found at {paths.candidate_results}; "
            "run stage 3 first"
        )
    selected = pl.read_parquet(SELECTED_SAMPLE)
    preprocessing_summary = pl.read_parquet(PREPROCESSING_SUMMARY)
    generations, task_settings, tasks = summarize_results(
        selected,
        candidate_results,
        preprocessing_summary,
    )

    candidate_results.write_parquet(paths.candidate_results)
    generations.write_parquet(paths.generation_results)
    task_settings.write_parquet(paths.task_setting_results)
    tasks.write_parquet(paths.task_results)
    complete_count = generations.get_column("evaluation_complete").sum()
    logger.info(
        "Summarized %d candidate results and %d/%d complete generations",
        candidate_results.height,
        complete_count,
        generations.height,
    )
    logger.info("Task summaries available for %d tasks", tasks.height)
    logger.info("Lowest observed task success rates:\n%s", tasks.head(10))
    logger.info("Highest observed task success rates:\n%s", tasks.tail(10))
    logger.info("Wrote task results to %s", paths.task_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
