#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import polars as pl

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import TextArtifact

_ROOT = Path(__file__).parents[1]
_DEFAULT_CORPUS = (
    _ROOT.parent / "gen-viewer" / "data" / "generation-corpus.parquet"
)
_REQUIRED_COLUMNS = frozenset({"decoder_output", "task_id"})


def _parquet_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Eagerly load the generation corpus and retain one HumanEval "
            "task's nonblank decoder outputs."
        )
    )
    parser.add_argument(
        "task_id", help="exact task ID, for example HumanEval/0"
    )
    parser.add_argument(
        "--parquet",
        type=_parquet_path,
        default=_DEFAULT_CORPUS,
        help=f"generation corpus path (default: {_DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help=(
            "run exhaustive preprocessing without caching after loading and "
            "filtering"
        ),
    )
    arguments = parser.parse_args()

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
    task_rows = corpus.filter(
        (pl.col("task_id") == arguments.task_id)
        & pl.col("decoder_output").is_not_null()
        & (pl.col("decoder_output").str.strip_chars() != "")
    )
    filter_seconds = perf_counter() - filter_started

    print(f"Parquet: {arguments.parquet}")
    print(f"Task ID: {arguments.task_id}")
    print(
        f"Loaded corpus: {corpus.height:,} rows x {corpus.width} columns "
        f"in {load_seconds:.6f} seconds"
    )
    print(
        f"Filtered rows: {task_rows.height:,} rows x {task_rows.width} "
        f"columns in {filter_seconds:.6f} seconds"
    )
    print(f"Filtered size: {task_rows.estimated_size('mb'):.3f} MB")
    print(
        f"Total load and filter: {load_seconds + filter_seconds:.6f} seconds"
    )

    if task_rows.is_empty():
        print(
            f"error: no nonblank decoder outputs found for task ID "
            f"{arguments.task_id!r}",
            file=sys.stderr,
        )
        return 1

    if arguments.preprocess:
        runner = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
        preprocessing_started = perf_counter()
        traces = [
            runner.run(TextArtifact(text=decoder_output))
            for decoder_output in task_rows.get_column(
                "decoder_output"
            ).to_list()
        ]
        preprocessing_seconds = perf_counter() - preprocessing_started
        print(
            f"Preprocessed without caching: {len(traces):,} traces in "
            f"{preprocessing_seconds:.6f} seconds "
            f"({len(traces) / preprocessing_seconds:,.2f} traces/second)"
        )
        print(
            "Total load, filter, and preprocessing: "
            f"{load_seconds + filter_seconds + preprocessing_seconds:.6f} "
            "seconds"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
