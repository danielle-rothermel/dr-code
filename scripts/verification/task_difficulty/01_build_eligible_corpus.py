#!/usr/bin/env python3

"""Build the preprocessing-eligible historical generation corpus."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import polars as pl
from dr_exec import ExecutionPoolConfig, FixedPoolCapacity
from dr_store import SqliteRecordCache

from dr_code.caching import (
    default_preprocess_batch_limits,
    preprocess_batch,
)
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
)
from dr_code.trace import (
    OUTPUT_KEY,
    Absent,
    InspectedCodeCandidateSetArtifact,
)

from corpus_loader import (
    format_manifest_summary,
    load_manifest_summary,
    load_workflow_frame,
)
from workflow_settings import (
    ELIGIBLE_CORPUS,
    EVALUATION_WORKERS,
    PREPROCESSING_CACHE,
    PREPROCESS_LOG,
    PREPROCESSING_SUMMARY,
    SETTINGS,
    _positive_worker_count,
    expected_manifest_sha256,
    generation_corpus_bundle_path,
    prepare_run_directory,
)

_REQUIRED_COLUMNS = frozenset(
    {
        "budget_mode",
        "decoder_model",
        "decoder_output",
        "encoder_model",
        "encoder_output",
        "encoder_user_prompt",
        "generation_mode",
        "max_characters",
        "model",
        "sample_id",
        "task_id",
    }
)


def _configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("task_difficulty.preprocess")
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


def _is_nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def classify_generation_rows(corpus: pl.DataFrame) -> pl.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(corpus.columns)
    if missing:
        raise ValueError(
            "generation corpus is missing required columns: "
            + ", ".join(sorted(missing))
        )

    filtered = corpus.filter(
        pl.struct(["generation_mode", "budget_mode"]).is_in(
            [
                {
                    "generation_mode": generation_mode,
                    "budget_mode": budget_mode,
                }
                for generation_mode, budget_mode in SETTINGS
            ]
        )
    )

    records: list[dict[str, object]] = []
    for row in filtered.iter_rows(named=True):
        decoder_output = row["decoder_output"]
        if not _is_nonblank(decoder_output):
            continue

        generation_mode = row["generation_mode"]
        if generation_mode == "enc_dec" and not _is_nonblank(
            row["encoder_output"]
        ):
            continue

        model_key = row["decoder_model"] or row["model"]
        if not _is_nonblank(model_key):
            raise ValueError(
                f"sample {row['sample_id']!r} has no decoder model"
            )

        records.append(
            {
                **row,
                "model_key": model_key,
            }
        )

    if not records:
        raise ValueError("no complete nonblank generation rows were found")
    return pl.DataFrame(records, infer_schema_length=None)


def _candidate_sources(trace_output: object) -> tuple[str, ...]:
    if isinstance(trace_output, Absent):
        return ()
    if not isinstance(trace_output, InspectedCodeCandidateSetArtifact):
        raise TypeError(
            "exhaustive preprocessing did not return inspected candidates"
        )
    sources: list[str] = []
    for inspected in trace_output.candidates:
        if not inspected.inspection.compiles:
            raise RuntimeError("final candidate does not compile")
        if not inspected.inspection.top_level_function_names:
            raise RuntimeError("final candidate has no top-level function")
        sources.append(inspected.candidate.source)
    return tuple(sources)


async def preprocess_distinct_outputs(
    outputs: Sequence[str],
    *,
    cache_path: Path,
    logger: logging.Logger,
    worker_count: int = EVALUATION_WORKERS,
) -> dict[str, tuple[str, ...]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")

    results: dict[str, tuple[str, ...]] = {}
    distinct_outputs = list(dict.fromkeys(outputs))
    started = perf_counter()
    failed_count = 0
    completed_count = 0
    logger.info(
        "Preprocessing %d distinct outputs with workers=%d",
        len(distinct_outputs),
        worker_count,
    )

    def _record_completion(_text: str, _trace: object) -> None:
        nonlocal completed_count
        completed_count += 1
        if completed_count % 500 == 0:
            logger.info(
                "Preprocessed %d distinct outputs in %.1f seconds",
                completed_count,
                perf_counter() - started,
            )

    async with await SqliteRecordCache.open(cache_path) as cache:
        traces = await preprocess_batch(
            distinct_outputs,
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            store=cache,
            pool_config=ExecutionPoolConfig(
                capacity=FixedPoolCapacity(max_active_jobs=worker_count)
            ),
            limits=default_preprocess_batch_limits(worker_count=worker_count),
            on_trace_completed=_record_completion,
        )

    for index, output in enumerate(distinct_outputs, start=1):
        trace = traces.get(output)
        if trace is None:
            failed_count += 1
            if failed_count <= 5:
                logger.warning(
                    "Preprocessing failed for distinct output %d/%d",
                    index,
                    len(distinct_outputs),
                )
            elif failed_count == 6:
                logger.warning(
                    "Further preprocessing failures will not be logged "
                    "individually"
                )
            results[output] = ()
            continue
        try:
            results[output] = _candidate_sources(trace.value(OUTPUT_KEY))
        except Exception as exc:
            failed_count += 1
            if failed_count <= 5:
                logger.warning(
                    "Preprocessing failed for distinct output %d/%d: %s: %s",
                    index,
                    len(distinct_outputs),
                    type(exc).__name__,
                    exc,
                )
            elif failed_count == 6:
                logger.warning(
                    "Further preprocessing failures will not be logged "
                    "individually"
                )
            results[output] = ()

    if failed_count:
        logger.info(
            "Preprocessing failed for %d/%d distinct outputs",
            failed_count,
            len(distinct_outputs),
        )
    return results


def attach_preprocessing_results(
    rows: pl.DataFrame,
    candidates_by_output: Mapping[str, tuple[str, ...]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    candidate_lists = [
        list(candidates_by_output[str(output)])
        for output in rows.get_column("decoder_output")
    ]
    measured = rows.with_columns(
        pl.Series("code_candidates", candidate_lists),
        pl.Series(
            "candidate_count",
            [len(candidates) for candidates in candidate_lists],
        ).cast(pl.Int64),
    ).with_columns(
        (pl.col("candidate_count") > 0).alias("preprocessing_succeeded")
    )
    summary = (
        measured.group_by(
            ["task_id", "generation_mode", "budget_mode", "model_key"]
        )
        .agg(
            pl.len().alias("nonblank_rows"),
            pl.col("preprocessing_succeeded").sum().alias("eligible_rows"),
            pl.col("candidate_count").sum().alias("candidate_count"),
            pl.col("decoder_output").n_unique().alias("distinct_outputs"),
        )
        .with_columns(
            (pl.col("eligible_rows") / pl.col("nonblank_rows")).alias(
                "preprocessing_success_rate"
            ),
            pl.lit(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID).alias(
                "preprocessing_definition_id"
            ),
            pl.lit(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION).alias(
                "preprocessing_definition_version"
            ),
        )
        .sort(["task_id", "generation_mode", "budget_mode", "model_key"])
    )
    eligible = (
        measured.filter(pl.col("preprocessing_succeeded"))
        .drop("preprocessing_succeeded")
        .with_columns(
            pl.lit(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID).alias(
                "preprocessing_definition_id"
            ),
            pl.lit(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION).alias(
                "preprocessing_definition_version"
            ),
        )
    )
    return eligible, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-bundle",
        type=Path,
        default=None,
        help=(
            "generation corpus bundle directory "
            f"(default: {generation_corpus_bundle_path()})"
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_worker_count,
        default=EVALUATION_WORKERS,
        help=(
            f"concurrent preprocessing workers (default: {EVALUATION_WORKERS})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    bundle_dir = (
        arguments.corpus_bundle.expanduser().resolve()
        if arguments.corpus_bundle is not None
        else generation_corpus_bundle_path()
    )
    prepare_run_directory()
    logger = _configure_logging(PREPROCESS_LOG)
    started = perf_counter()
    manifest_summary = load_manifest_summary(bundle_dir)
    logger.info(
        "Loading generation corpus bundle %s (%s)",
        bundle_dir,
        format_manifest_summary(manifest_summary),
    )
    corpus = load_workflow_frame(
        bundle_dir,
        expected_manifest_sha256=expected_manifest_sha256(),
    )
    logger.info("Loaded %d rows and %d columns", corpus.height, corpus.width)

    rows = classify_generation_rows(corpus)
    outputs: list[str] = rows.get_column("decoder_output").to_list()
    logger.info(
        "Classified %d complete nonblank rows with %d distinct outputs",
        rows.height,
        len(set(outputs)),
    )
    candidates = asyncio.run(
        preprocess_distinct_outputs(
            outputs,
            cache_path=PREPROCESSING_CACHE,
            logger=logger,
            worker_count=arguments.workers,
        )
    )
    eligible, summary = attach_preprocessing_results(rows, candidates)
    eligible.write_parquet(ELIGIBLE_CORPUS)
    summary.write_parquet(PREPROCESSING_SUMMARY)
    logger.info(
        "Wrote %d eligible rows (%.2f%%) to %s",
        eligible.height,
        100 * eligible.height / rows.height,
        ELIGIBLE_CORPUS,
    )
    logger.info(
        "Wrote preprocessing denominators to %s", PREPROCESSING_SUMMARY
    )
    logger.info("Finished in %.1f seconds", perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
