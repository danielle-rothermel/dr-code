#!/usr/bin/env python3

"""Build the preprocessing-eligible historical generation corpus."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import polars as pl
from dr_store import SqliteRecordCache

from dr_code.caching import run_preprocessing_cached
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_VERSION,
    bind_preprocessing,
)
from dr_code.trace import (
    OUTPUT_KEY,
    Absent,
    InspectedCodeCandidateSetArtifact,
)

from workflow_settings import (
    ELIGIBLE_CORPUS,
    GENERATION_CORPUS,
    PREPROCESSING_CACHE,
    PREPROCESS_LOG,
    PREPROCESSING_SUMMARY,
    SETTINGS,
    SOURCE_KIND,
    prepare_run_directory,
)

_REQUIRED_COLUMNS = frozenset(
    {
        "decoder_model",
        "decoder_output",
        "encoder_model",
        "encoder_output",
        "encoder_user_prompt",
        "model",
        "sample_id",
        "source_kind",
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


def _max_characters(value: object) -> int | None:
    if not _is_nonblank(value):
        return None
    assert isinstance(value, str)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        if "max_characters" in value:
            raise ValueError(
                "encoder_user_prompt mentions max_characters but is not "
                "valid JSON"
            ) from exc
        return None
    if not isinstance(payload, Mapping) or "max_characters" not in payload:
        return None
    maximum = payload["max_characters"]
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < 1
    ):
        raise ValueError("max_characters must be a positive integer")
    return maximum


def classify_generation_rows(corpus: pl.DataFrame) -> pl.DataFrame:
    missing = _REQUIRED_COLUMNS.difference(corpus.columns)
    if missing:
        raise ValueError(
            "generation corpus is missing required columns: "
            + ", ".join(sorted(missing))
        )

    records: list[dict[str, object]] = []
    for row in corpus.filter(pl.col("source_kind") == SOURCE_KIND).iter_rows(
        named=True
    ):
        decoder_output = row["decoder_output"]
        if not _is_nonblank(decoder_output):
            continue

        encoder_model = row["encoder_model"]
        generation_mode = (
            "enc_dec" if _is_nonblank(encoder_model) else "direct"
        )
        if generation_mode == "enc_dec" and not _is_nonblank(
            row["encoder_output"]
        ):
            continue

        model_key = row["decoder_model"] or row["model"]
        if not _is_nonblank(model_key):
            raise ValueError(
                f"sample {row['sample_id']!r} has no decoder model"
            )
        maximum = _max_characters(row["encoder_user_prompt"])
        budget_mode = "budget" if maximum is not None else "no_budget"
        setting = (generation_mode, budget_mode)
        if setting not in SETTINGS:
            raise ValueError(f"unexpected experiment setting: {setting!r}")

        records.append(
            {
                **row,
                "generation_mode": generation_mode,
                "budget_mode": budget_mode,
                "max_characters": maximum,
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


def preprocess_distinct_outputs(
    outputs: Sequence[str],
    *,
    cache_path: Path,
    logger: logging.Logger,
) -> dict[str, tuple[str, ...]]:
    runner = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    results: dict[str, tuple[str, ...]] = {}
    started = perf_counter()
    with SqliteRecordCache(cache_path) as cache:
        for index, output in enumerate(dict.fromkeys(outputs), start=1):
            trace = run_preprocessing_cached(output, runner, cache)
            results[output] = _candidate_sources(trace.value(OUTPUT_KEY))
            if index % 500 == 0:
                logger.info(
                    "Preprocessed %d distinct outputs in %.1f seconds",
                    index,
                    perf_counter() - started,
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


def main() -> int:
    prepare_run_directory()
    logger = _configure_logging(PREPROCESS_LOG)
    started = perf_counter()
    logger.info("Loading %s", GENERATION_CORPUS)
    corpus = pl.read_parquet(GENERATION_CORPUS)
    logger.info("Loaded %d rows and %d columns", corpus.height, corpus.width)

    rows = classify_generation_rows(corpus)
    outputs: list[str] = rows.get_column("decoder_output").to_list()
    logger.info(
        "Classified %d complete nonblank rows with %d distinct outputs",
        rows.height,
        len(set(outputs)),
    )
    candidates = preprocess_distinct_outputs(
        outputs,
        cache_path=PREPROCESSING_CACHE,
        logger=logger,
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
