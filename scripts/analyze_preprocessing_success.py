#!/usr/bin/env python3

"""Estimate exhaustive-preprocessing success over sampled HumanEval tasks."""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final

import polars as pl

from bootstrap_statistics import (
    BootstrapConfidenceInterval,
    bootstrap_confidence_interval,
)
from dr_code.caching import preprocess_batch
from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
from dr_code.trace import (
    OUTPUT_KEY,
    Absent,
    InspectedCodeCandidateSetArtifact,
    Trace,
)

_ROOT = Path(__file__).parents[1]
_DEFAULT_CORPUS = (
    _ROOT.parent / "gen-viewer" / "data" / "generation-corpus.parquet"
)
_REQUIRED_COLUMNS = frozenset({"decoder_output", "sample_id", "task_id"})
_DEFAULT_WORKERS: Final = 16


@dataclass(frozen=True, slots=True)
class _TaskPreprocessingResult:
    task_id: str
    rows: int
    distinct_outputs: int
    successful_rows: int
    failure_codes: tuple[tuple[str, int], ...]
    preprocessing_seconds: float

    @property
    def failed_rows(self) -> int:
        return self.rows - self.successful_rows

    @property
    def success_rate(self) -> float:
        return self.successful_rows / self.rows

    @property
    def traces_per_second(self) -> float:
        return self.rows / self.preprocessing_seconds


def _parquet_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _failure_code(trace: Trace) -> str | None:
    output = trace.value(OUTPUT_KEY)
    if isinstance(output, Absent):
        return output.failure_code
    if not isinstance(output, InspectedCodeCandidateSetArtifact):
        raise TypeError(
            "preprocessing output must be inspected candidates or absent"
        )
    return None


def _analyze_task(
    task_id: str,
    task_rows: pl.DataFrame,
    *,
    worker_count: int,
) -> _TaskPreprocessingResult:
    decoder_outputs: list[str] = task_rows.get_column(
        "decoder_output"
    ).to_list()
    started = perf_counter()
    traces_by_text = asyncio.run(
        preprocess_batch(
            decoder_outputs,
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            worker_count=worker_count,
        )
    )
    preprocessing_seconds = perf_counter() - started
    failures: Counter[str] = Counter()
    successful_rows = 0
    for text in decoder_outputs:
        trace = traces_by_text.get(text)
        if trace is None:
            failures["preprocess_execution_failed"] += 1
            continue
        failure_code = _failure_code(trace)
        if failure_code is None:
            successful_rows += 1
        else:
            failures[failure_code] += 1
    return _TaskPreprocessingResult(
        task_id=task_id,
        rows=len(decoder_outputs),
        distinct_outputs=len(set(decoder_outputs)),
        successful_rows=successful_rows,
        failure_codes=tuple(sorted(failures.items())),
        preprocessing_seconds=preprocessing_seconds,
    )


def _aggregate_success_rate(
    results: Sequence[_TaskPreprocessingResult],
) -> float:
    successful_rows = sum(result.successful_rows for result in results)
    rows = sum(result.rows for result in results)
    return successful_rows / rows


def _combined_failure_codes(
    results: Sequence[_TaskPreprocessingResult],
) -> tuple[tuple[str, int], ...]:
    failures: Counter[str] = Counter()
    for result in results:
        failures.update(dict(result.failure_codes))
    return tuple(sorted(failures.items()))


def _print_failure_codes(
    failure_codes: Sequence[tuple[str, int]],
    *,
    failed_rows: int,
) -> None:
    print("Preprocessing failures:")
    if not failure_codes:
        print("  none")
        return
    for failure_code, count in failure_codes:
        print(
            f"  {failure_code}: {count:,} "
            f"({count / failed_rows:.2%} of failures)"
        )


def _print_interval(
    label: str,
    interval: BootstrapConfidenceInterval,
) -> None:
    confidence = interval.confidence_level * 100
    print(
        f"{label}: {interval.estimate:.2%} "
        f"({confidence:g}% bootstrap CI "
        f"{interval.lower:.2%} to {interval.upper:.2%})"
    )


def _print_single_task(result: _TaskPreprocessingResult) -> None:
    print(f"Filtered rows: {result.rows:,}")
    print(f"Distinct decoder outputs: {result.distinct_outputs:,}")
    print(
        f"Preprocessing success: {result.success_rate:.2%} "
        f"({result.successful_rows:,}/{result.rows:,} nonblank samples)"
    )
    print(
        f"Full preprocessing: {result.preprocessing_seconds:.6f} seconds "
        f"({result.traces_per_second:,.2f} traces/second)"
    )
    _print_failure_codes(
        result.failure_codes,
        failed_rows=result.failed_rows,
    )


def _print_task_sample(
    results: Sequence[_TaskPreprocessingResult],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> None:
    print(
        f"{'task':18} {'rows':>8} {'success':>8} {'success rate':>13} "
        f"{'failed':>8} {'distinct':>9} {'preprocess':>11} {'traces/s':>11}"
    )
    for result in results:
        print(
            f"{result.task_id:18} {result.rows:8,d} "
            f"{result.successful_rows:8,d} {result.success_rate:12.2%} "
            f"{result.failed_rows:8,d} {result.distinct_outputs:9,d} "
            f"{result.preprocessing_seconds:10.3f}s "
            f"{result.traces_per_second:11,.2f}"
        )

    print()
    print("Per-task preprocessing failure codes:")
    for result in results:
        failures = ", ".join(
            f"{failure_code}={count:,}"
            for failure_code, count in result.failure_codes
        )
        print(f"  {result.task_id}: {failures or 'none'}")

    total_rows = sum(result.rows for result in results)
    total_successes = sum(result.successful_rows for result in results)
    total_failures = total_rows - total_successes
    print()
    print(
        f"Aggregate preprocessing success: "
        f"{total_successes / total_rows:.2%} "
        f"({total_successes:,}/{total_rows:,} nonblank samples)"
    )
    _print_failure_codes(
        _combined_failure_codes(results),
        failed_rows=total_failures,
    )

    interval = bootstrap_confidence_interval(
        results,
        _aggregate_success_rate,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    print()
    _print_interval("Aggregate preprocessing success rate", interval)
    print(
        f"Bootstrap unit: task; resamples: {bootstrap_resamples:,}; seed: {seed}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate exhaustive-preprocessing success over selected "
            "HumanEval generation tasks by running every row directly "
            "without candidate evaluation."
        )
    )
    parser.add_argument(
        "task_id",
        nargs="?",
        help="exact task ID, for example HumanEval/0",
    )
    parser.add_argument(
        "--task-count",
        type=_positive_int,
        help="number of task IDs to sample instead of one task ID",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for task selection and bootstrap resampling",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=_positive_int,
        default=10_000,
        help="number of task-level bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="bootstrap confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=_DEFAULT_WORKERS,
        help=f"concurrent preprocessing workers (default: {_DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--parquet",
        type=_parquet_path,
        default=_DEFAULT_CORPUS,
        help=f"generation corpus path (default: {_DEFAULT_CORPUS})",
    )
    arguments = parser.parse_args()
    if (arguments.task_id is None) == (arguments.task_count is None):
        parser.error("provide exactly one task_id or --task-count")
    if arguments.task_count is not None and arguments.task_count < 2:
        parser.error("--task-count must be at least 2 for bootstrapping")
    if not 0.0 < arguments.confidence_level < 1.0:
        parser.error("--confidence-level must be between zero and one")

    load_started = perf_counter()
    corpus = pl.read_parquet(arguments.parquet)
    load_seconds = perf_counter() - load_started
    missing_columns = _REQUIRED_COLUMNS.difference(corpus.columns)
    if missing_columns:
        parser.error(
            "parquet is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    filter_started = perf_counter()
    nonblank_rows = corpus.filter(
        pl.col("task_id").is_not_null()
        & pl.col("decoder_output").is_not_null()
        & (pl.col("decoder_output").str.strip_chars() != "")
    )
    if arguments.task_count is None:
        assert arguments.task_id is not None
        selected_task_ids: list[str] = [arguments.task_id]
    else:
        available_task_ids: list[str] = (
            nonblank_rows.get_column("task_id").unique().sort().to_list()
        )
        if arguments.task_count > len(available_task_ids):
            parser.error(
                f"--task-count {arguments.task_count} exceeds "
                f"{len(available_task_ids)} eligible tasks"
            )
        selected_task_ids = random.Random(arguments.seed).sample(
            available_task_ids,
            arguments.task_count,
        )
    selected_rows = nonblank_rows.filter(
        pl.col("task_id").is_in(selected_task_ids)
    )
    filter_seconds = perf_counter() - filter_started
    if selected_rows.is_empty():
        print(
            f"error: no nonblank decoder outputs found for "
            f"{selected_task_ids!r}",
            file=sys.stderr,
        )
        return 1

    results: list[_TaskPreprocessingResult] = []
    for index, task_id in enumerate(selected_task_ids, start=1):
        task_rows = selected_rows.filter(pl.col("task_id") == task_id)
        print(
            f"Analyzing task {index}/{len(selected_task_ids)}: "
            f"{task_id} ({task_rows.height:,} rows)",
            flush=True,
        )
        results.append(
            _analyze_task(task_id, task_rows, worker_count=arguments.workers)
        )

    print(f"Parquet: {arguments.parquet}")
    print(
        f"Loaded corpus: {corpus.height:,} rows x {corpus.width} columns "
        f"in {load_seconds:.6f} seconds"
    )
    print(
        f"Selected and filtered: {selected_rows.height:,} rows in "
        f"{filter_seconds:.6f} seconds"
    )
    if arguments.task_count is None:
        print(f"Task ID: {results[0].task_id}")
        _print_single_task(results[0])
    else:
        print(
            f"Randomly selected task IDs (seed {arguments.seed}): "
            + ", ".join(selected_task_ids)
        )
        _print_task_sample(
            results,
            confidence_level=arguments.confidence_level,
            bootstrap_resamples=arguments.bootstrap_resamples,
            seed=arguments.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
