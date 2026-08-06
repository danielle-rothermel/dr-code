#!/usr/bin/env python3

"""Benchmark an experimental in-memory per-step preprocessing cache."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import polars as pl

from bootstrap_statistics import (
    BootstrapConfidenceInterval,
    bootstrap_confidence_interval,
)
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


@dataclass(frozen=True, slots=True)
class _TaskBenchmark:
    task_id: str
    rows: int
    distinct_outputs: int
    uncached_seconds: float
    cold_seconds: float
    warm_seconds: float
    cold_stats: tuple[_StepStats, ...]
    warm_stats: tuple[_StepStats, ...]

    @property
    def cold_speedup(self) -> float:
        return self.uncached_seconds / self.cold_seconds


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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


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


def _benchmark_task(task_id: str, task_rows: pl.DataFrame) -> _TaskBenchmark:
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
    return _TaskBenchmark(
        task_id=task_id,
        rows=len(decoder_outputs),
        distinct_outputs=len(set(decoder_outputs)),
        uncached_seconds=baseline_seconds,
        cold_seconds=cold_seconds,
        warm_seconds=warm_seconds,
        cold_stats=cold_stats,
        warm_stats=warm_stats,
    )


def _aggregate_cold_hit_rate(results: Sequence[_TaskBenchmark]) -> float:
    hits = 0
    lookups = 0
    for result in results:
        result_hits, result_lookups, _ = _hit_rate(result.cold_stats)
        hits += result_hits
        lookups += result_lookups
    return hits / lookups


def _aggregate_cold_speedup(results: Sequence[_TaskBenchmark]) -> float:
    uncached_seconds = sum(result.uncached_seconds for result in results)
    cold_seconds = sum(result.cold_seconds for result in results)
    return uncached_seconds / cold_seconds


def _print_interval(
    label: str,
    interval: BootstrapConfidenceInterval,
    *,
    percentage: bool,
) -> None:
    confidence = f"{interval.confidence_level:.1%}"
    if percentage:
        estimate = f"{interval.estimate:.2%}"
        lower = f"{interval.lower:.2%}"
        upper = f"{interval.upper:.2%}"
    else:
        estimate = f"{interval.estimate:.2f}x"
        lower = f"{interval.lower:.2f}x"
        upper = f"{interval.upper:.2f}x"
    print(
        f"{label}: {estimate} ({confidence} bootstrap CI: {lower} to {upper})"
    )


def _print_single_task(result: _TaskBenchmark) -> None:
    print(f"Filtered rows: {result.rows:,}")
    print(f"Distinct decoder outputs: {result.distinct_outputs:,}")
    print(
        f"Uncached baseline: {result.uncached_seconds:.6f} seconds "
        f"({result.rows / result.uncached_seconds:,.2f} traces/second)"
    )
    print(
        f"Cold per-step cache: {result.cold_seconds:.6f} seconds "
        f"({result.cold_speedup:.2f}x baseline)"
    )
    print(
        f"Warm per-step cache: {result.warm_seconds:.6f} seconds "
        f"({result.uncached_seconds / result.warm_seconds:.2f}x baseline)"
    )
    print("Trace equivalence: exact for cold and warm cache runs")
    print()
    _print_step_stats("Cold per-step cache", result.cold_stats)
    print()
    _print_step_stats("Warm per-step cache", result.warm_stats)


def _print_task_sample(
    results: Sequence[_TaskBenchmark],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> None:
    print(
        f"{'task':18} {'rows':>8} {'distinct':>9} {'hit rate':>10} "
        f"{'uncached':>10} {'cold':>10} {'speedup':>9}"
    )
    for result in results:
        _, _, hit_rate = _hit_rate(result.cold_stats)
        print(
            f"{result.task_id:18} {result.rows:8,d} "
            f"{result.distinct_outputs:9,d} {hit_rate:9.2%} "
            f"{result.uncached_seconds:9.3f}s "
            f"{result.cold_seconds:9.3f}s {result.cold_speedup:8.2f}x"
        )

    hit_rate_interval = bootstrap_confidence_interval(
        results,
        _aggregate_cold_hit_rate,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    speedup_interval = bootstrap_confidence_interval(
        results,
        _aggregate_cold_speedup,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    print()
    _print_interval(
        "Aggregate cold-cache hit rate",
        hit_rate_interval,
        percentage=True,
    )
    _print_interval(
        "Aggregate uncached-to-cold speedup",
        speedup_interval,
        percentage=False,
    )
    print(
        f"Bootstrap unit: task; resamples: {bootstrap_resamples:,}; "
        f"seed: {seed}"
    )
    print("Trace equivalence: exact for every selected task")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exhaustive preprocessing with an experimental "
            "in-memory per-step cache."
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
        help="number of task IDs to select randomly instead of one task ID",
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
    eligible_rows = corpus.filter(
        pl.col("task_id").is_not_null()
        & pl.col("decoder_output").is_not_null()
        & (pl.col("decoder_output").str.strip_chars() != "")
    )
    if arguments.task_count is None:
        assert arguments.task_id is not None
        selected_task_ids: list[str] = [arguments.task_id]
    else:
        available_task_ids: list[str] = (
            eligible_rows.get_column("task_id").unique().sort().to_list()
        )
        if arguments.task_count > len(available_task_ids):
            parser.error(
                f"--task-count {arguments.task_count} exceeds "
                f"{len(available_task_ids)} eligible tasks"
            )
        selected_task_ids = random.Random(arguments.seed).sample(
            available_task_ids, arguments.task_count
        )
    selected_rows = eligible_rows.filter(
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

    results: list[_TaskBenchmark] = []
    for index, task_id in enumerate(selected_task_ids, start=1):
        task_rows = selected_rows.filter(pl.col("task_id") == task_id)
        print(
            f"Benchmarking task {index}/{len(selected_task_ids)}: "
            f"{task_id} ({task_rows.height:,} rows)",
            flush=True,
        )
        results.append(_benchmark_task(task_id, task_rows))

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
