#!/usr/bin/env python3

"""Benchmark an experimental in-memory per-step preprocessing cache."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    BoundPreprocessingRunner,
    bind_preprocessing,
)
from dr_code.preprocessing.steps.base import StepFailedError
from dr_code.trace import (
    INPUT_KEY,
    OUTPUT_KEY,
    Absent,
    Artifact,
    JsonFactValue,
    TextArtifact,
    Trace,
    is_absent,
    serialize_trace,
)

_ROOT = Path(__file__).parents[1]
_DEFAULT_CORPUS = (
    _ROOT.parent / "gen-viewer" / "data" / "generation-corpus.parquet"
)
_REQUIRED_COLUMNS = frozenset({"decoder_output", "sample_id", "task_id"})


@dataclass(frozen=True, slots=True)
class _CachedStepResult:
    value: Artifact | Absent
    facts: dict[str, JsonFactValue]


@dataclass(slots=True)
class _MutableStepStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    skipped_absent: int = 0


@dataclass(frozen=True, slots=True)
class _StepStats:
    instance_name: str
    lookups: int
    hits: int
    misses: int
    skipped_absent: int
    entries: int


class _InMemoryStepCacheRunner:
    """Experimental runner mirroring the production runner's semantics."""

    def __init__(self, runner: BoundPreprocessingRunner) -> None:
        self._runner = runner
        self._caches: dict[str, dict[Artifact, _CachedStepResult]] = {
            bound.instance_name: {} for bound in runner.steps
        }
        self._stats = self._new_stats()

    def _new_stats(self) -> dict[str, _MutableStepStats]:
        return {
            bound.instance_name: _MutableStepStats()
            for bound in self._runner.steps
        }

    def reset_stats(self) -> None:
        self._stats = self._new_stats()

    def stats(self) -> tuple[_StepStats, ...]:
        return tuple(
            _StepStats(
                instance_name=bound.instance_name,
                lookups=self._stats[bound.instance_name].lookups,
                hits=self._stats[bound.instance_name].hits,
                misses=self._stats[bound.instance_name].misses,
                skipped_absent=(
                    self._stats[bound.instance_name].skipped_absent
                ),
                entries=len(self._caches[bound.instance_name]),
            )
            for bound in self._runner.steps
        )

    def run(self, input_value: TextArtifact) -> Trace:
        values: dict[str, Artifact | Absent] = {INPUT_KEY: input_value}
        step_facts: dict[str, dict[str, JsonFactValue]] = {}
        current: Artifact | Absent = input_value

        # Bound step internals are used only by this isolated experiment.
        for bound in self._runner.steps:
            stats = self._stats[bound.instance_name]
            if is_absent(current):
                stats.skipped_absent += 1
                current = Absent(
                    failed_step=current.failed_step,
                    failure_code=current.failure_code,
                    cause=current.cause,
                    propagated_through=(
                        *current.propagated_through,
                        bound.instance_name,
                    ),
                )
            else:
                stats.lookups += 1
                cache = self._caches[bound.instance_name]
                result = cache.get(current)
                if result is None:
                    stats.misses += 1
                    try:
                        output = bound.step.apply(current)
                    except StepFailedError as exc:
                        result = _CachedStepResult(
                            value=Absent(
                                failed_step=bound.instance_name,
                                failure_code=exc.code.value,
                                cause=exc.cause,
                            ),
                            facts=dict(exc.evidence),
                        )
                    else:
                        result = _CachedStepResult(
                            value=output.value,
                            facts=dict(output.facts),
                        )
                    cache[current] = result
                else:
                    stats.hits += 1
                current = result.value
                if result.facts:
                    step_facts[bound.instance_name] = dict(result.facts)
            values[bound.instance_name] = current

        values[OUTPUT_KEY] = current
        return Trace(
            values=values,
            producer=self._runner.producer,
            step_facts=step_facts,
        )


def _parquet_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def _assert_same_traces(
    *,
    label: str,
    sample_ids: list[str],
    expected: list[Trace],
    actual: list[Trace],
) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(
            f"{label} produced {len(actual)} traces; expected {len(expected)}"
        )
    for sample_id, expected_trace, actual_trace in zip(
        sample_ids, expected, actual, strict=True
    ):
        if serialize_trace(actual_trace) != serialize_trace(expected_trace):
            raise RuntimeError(
                f"{label} trace differs for sample_id {sample_id!r}"
            )


def _hit_rate(stats: tuple[_StepStats, ...]) -> tuple[int, int, float]:
    hits = sum(step.hits for step in stats)
    lookups = sum(step.lookups for step in stats)
    rate = hits / lookups if lookups else 0.0
    return hits, lookups, rate


def _print_step_stats(label: str, stats: tuple[_StepStats, ...]) -> None:
    hits, lookups, rate = _hit_rate(stats)
    print(f"{label} hit rate: {rate:.2%} ({hits:,}/{lookups:,})")
    print(
        f"{'step':32} {'lookups':>9} {'hits':>9} {'misses':>9} "
        f"{'hit rate':>10} {'skipped':>9} {'entries':>9}"
    )
    for step in stats:
        step_rate = step.hits / step.lookups if step.lookups else 0.0
        print(
            f"{step.instance_name:32} {step.lookups:9,d} "
            f"{step.hits:9,d} {step.misses:9,d} {step_rate:9.2%} "
            f"{step.skipped_absent:9,d} {step.entries:9,d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exhaustive preprocessing with an experimental "
            "in-memory per-step cache."
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
    if task_rows.is_empty():
        print(
            f"error: no nonblank decoder outputs found for task ID "
            f"{arguments.task_id!r}",
            file=sys.stderr,
        )
        return 1

    sample_ids: list[str] = task_rows.get_column("sample_id").to_list()
    decoder_outputs: list[str] = task_rows.get_column(
        "decoder_output"
    ).to_list()

    baseline_runner = bind_preprocessing(
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION
    )
    baseline_started = perf_counter()
    baseline_traces = [
        baseline_runner.run(TextArtifact(text=text))
        for text in decoder_outputs
    ]
    baseline_seconds = perf_counter() - baseline_started

    cached_runner = _InMemoryStepCacheRunner(
        bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    )
    cold_started = perf_counter()
    cold_traces = [
        cached_runner.run(TextArtifact(text=text)) for text in decoder_outputs
    ]
    cold_seconds = perf_counter() - cold_started
    cold_stats = cached_runner.stats()
    _assert_same_traces(
        label="cold per-step cache",
        sample_ids=sample_ids,
        expected=baseline_traces,
        actual=cold_traces,
    )

    cached_runner.reset_stats()
    warm_started = perf_counter()
    warm_traces = [
        cached_runner.run(TextArtifact(text=text)) for text in decoder_outputs
    ]
    warm_seconds = perf_counter() - warm_started
    warm_stats = cached_runner.stats()
    _assert_same_traces(
        label="warm per-step cache",
        sample_ids=sample_ids,
        expected=baseline_traces,
        actual=warm_traces,
    )

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
    print(f"Distinct decoder outputs: {len(set(decoder_outputs)):,}")
    print(
        f"Uncached baseline: {baseline_seconds:.6f} seconds "
        f"({len(decoder_outputs) / baseline_seconds:,.2f} traces/second)"
    )
    print(
        f"Cold per-step cache: {cold_seconds:.6f} seconds "
        f"({baseline_seconds / cold_seconds:.2f}x baseline)"
    )
    print(
        f"Warm per-step cache: {warm_seconds:.6f} seconds "
        f"({baseline_seconds / warm_seconds:.2f}x baseline)"
    )
    print("Trace equivalence: exact for cold and warm cache runs")
    print()
    _print_step_stats("Cold per-step cache", cold_stats)
    print()
    _print_step_stats("Warm per-step cache", warm_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
