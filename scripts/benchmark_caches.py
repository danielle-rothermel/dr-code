#!/usr/bin/env python3

"""Benchmark full preprocessing followed by test-result cache lookup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
from time import perf_counter

import polars as pl

from bootstrap_statistics import (
    BootstrapConfidenceInterval,
    bootstrap_confidence_interval,
)
from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.metric_operator import CodeTest, CodeTestSettings
from dr_code.humaneval.sampling import load_humaneval_rows
from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    InMemoryExecutionCache,
)
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    StepName,
    bind_preprocessing,
)
from dr_code.trace import (
    OUTPUT_KEY,
    Absent,
    CodeArtifact,
    CodeCandidateSetArtifact,
    InspectedCodeCandidateSetArtifact,
    JsonArtifact,
    TextArtifact,
    Trace,
)

_ROOT = Path(__file__).parents[1]
_DEFAULT_CORPUS = (
    _ROOT.parent / "gen-viewer" / "data" / "generation-corpus.parquet"
)
_DEFAULT_HUMANEVAL_SNAPSHOT = (
    _ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
)
_REQUIRED_COLUMNS = frozenset({"decoder_output", "sample_id", "task_id"})
_RAW_CANDIDATE_KEY = StepName.EXTRACT_ALL_REPRESENTATIONS.value
_DERIVED_TASK_FIELDS: frozenset[str] = frozenset(
    {"parsed", "parsed_tests", *HumanEvalTask.model_computed_fields}
)


@dataclass(frozen=True, slots=True)
class _ReuseStats:
    occurrences: int
    unique_keys: int

    @property
    def hits(self) -> int:
        return self.occurrences - self.unique_keys

    @property
    def hit_rate(self) -> float:
        return self.hits / self.occurrences if self.occurrences else 0.0


@dataclass(frozen=True, slots=True)
class _RowCacheResult:
    failure_code: str | None
    candidate_count: int
    candidate_hits: int

    def __post_init__(self) -> None:
        if self.failure_code is not None:
            if self.candidate_count != 0 or self.candidate_hits != 0:
                raise ValueError("failed rows cannot contain candidates")
            return
        if self.candidate_count < 1:
            raise ValueError("successful rows must contain a candidate")
        if not 0 <= self.candidate_hits <= self.candidate_count:
            raise ValueError("candidate hits must not exceed candidates")

    @property
    def succeeded(self) -> bool:
        return self.failure_code is None

    @property
    def uncached_candidates(self) -> int:
        return self.candidate_count - self.candidate_hits


@dataclass(frozen=True, slots=True)
class _UncachedCandidateDistribution:
    total: int
    mean: float
    median: float
    p95: int


@dataclass(frozen=True, slots=True)
class _RowCacheSummary:
    rows: int
    failed_rows: int
    successful_rows: int
    zero_hit_rows: int
    partial_hit_rows: int
    fully_cached_rows: int
    candidate_count: int
    candidate_hits: int
    uncached_candidates: tuple[int, ...]
    failure_codes: tuple[tuple[str, int], ...]

    @property
    def candidate_hit_rate(self) -> float:
        return (
            self.candidate_hits / self.candidate_count
            if self.candidate_count
            else 0.0
        )

    @property
    def overall_full_skip_rate(self) -> float:
        return self.fully_cached_rows / self.rows if self.rows else 0.0

    @property
    def conditional_full_skip_rate(self) -> float:
        return (
            self.fully_cached_rows / self.successful_rows
            if self.successful_rows
            else 0.0
        )

    @property
    def uncached_candidate_distribution(
        self,
    ) -> _UncachedCandidateDistribution:
        values = sorted(self.uncached_candidates)
        if not values:
            return _UncachedCandidateDistribution(
                total=0,
                mean=0.0,
                median=0.0,
                p95=0,
            )
        p95_index = math.ceil(0.95 * len(values)) - 1
        return _UncachedCandidateDistribution(
            total=sum(values),
            mean=fmean(values),
            median=median(values),
            p95=values[p95_index],
        )


@dataclass(frozen=True, slots=True)
class _CandidateMeasurements:
    raw_source_keys: tuple[str, ...]
    final_source_keys: tuple[str, ...]
    execution_request_keys: tuple[str, ...]
    row_cache: _RowCacheSummary


@dataclass(frozen=True, slots=True)
class _TaskBenchmark:
    task_id: str
    rows: int
    distinct_outputs: int
    preprocessing_seconds: float
    test_cache_seconds: float
    candidates: _CandidateMeasurements

    @property
    def combined_seconds(self) -> float:
        return self.preprocessing_seconds + self.test_cache_seconds

    @property
    def test_cache_added_fraction(self) -> float:
        return self.test_cache_seconds / self.preprocessing_seconds

    @property
    def combined_factor(self) -> float:
        return self.combined_seconds / self.preprocessing_seconds

    @property
    def success_rate(self) -> float:
        return self.row_cache.successful_rows / self.rows

    @property
    def row_cache(self) -> _RowCacheSummary:
        return self.candidates.row_cache

    @property
    def raw_candidate_reuse(self) -> _ReuseStats:
        return _reuse_stats(self.candidates.raw_source_keys)

    @property
    def final_candidate_reuse(self) -> _ReuseStats:
        return _reuse_stats(self.candidates.final_source_keys)

    @property
    def execution_request_reuse(self) -> _ReuseStats:
        return _reuse_stats(self.candidates.execution_request_keys)


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


def _reuse_stats(keys: Iterable[str]) -> _ReuseStats:
    occurrences = 0
    unique_keys: set[str] = set()
    for key in keys:
        occurrences += 1
        unique_keys.add(key)
    return _ReuseStats(
        occurrences=occurrences,
        unique_keys=len(unique_keys),
    )


def _execution_request_key(request: ExecutionRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summarize_row_cache(
    rows: Sequence[_RowCacheResult],
) -> _RowCacheSummary:
    failures = Counter(
        row.failure_code for row in rows if row.failure_code is not None
    )
    uncached_candidates = tuple(
        row.uncached_candidates for row in rows if row.succeeded
    )
    return _RowCacheSummary(
        rows=len(rows),
        failed_rows=sum(not row.succeeded for row in rows),
        successful_rows=sum(row.succeeded for row in rows),
        zero_hit_rows=sum(
            row.succeeded and row.candidate_hits == 0 for row in rows
        ),
        partial_hit_rows=sum(
            0 < row.candidate_hits < row.candidate_count for row in rows
        ),
        fully_cached_rows=sum(
            row.succeeded and row.candidate_hits == row.candidate_count
            for row in rows
        ),
        candidate_count=sum(row.candidate_count for row in rows),
        candidate_hits=sum(row.candidate_hits for row in rows),
        uncached_candidates=uncached_candidates,
        failure_codes=tuple(sorted(failures.items())),
    )


def _candidate_measurements(
    traces: Sequence[Trace],
    task: HumanEvalTask,
) -> _CandidateMeasurements:
    raw_source_keys: list[str] = []
    final_source_keys: list[str] = []
    execution_request_keys: list[str] = []
    row_results: list[_RowCacheResult] = []
    execution_cache = InMemoryExecutionCache()
    placeholder_outcome = ExecutionOutcome(returncode=0, stdout="", stderr="")
    code_test = CodeTest(CodeTestSettings())
    task_artifact = JsonArtifact(
        payload=task.model_dump(
            mode="json",
            exclude=set(_DERIVED_TASK_FIELDS),
        )
    )

    for trace in traces:
        raw_candidates = trace.value(_RAW_CANDIDATE_KEY)
        if isinstance(raw_candidates, CodeCandidateSetArtifact):
            raw_source_keys.extend(
                candidate.source for candidate in raw_candidates.candidates
            )

        final_candidates = trace.value(OUTPUT_KEY)
        if isinstance(final_candidates, Absent):
            row_results.append(
                _RowCacheResult(
                    failure_code=final_candidates.failure_code,
                    candidate_count=0,
                    candidate_hits=0,
                )
            )
            continue
        if not isinstance(final_candidates, InspectedCodeCandidateSetArtifact):
            raise TypeError(
                "preprocessing output must be inspected candidates or absent"
            )

        row_request_keys: list[str] = []
        row_missed_request_keys: set[str] = set()
        candidate_hits = 0
        # Compare the whole row with cache state from earlier rows. Updating
        # afterward prevents work found inside this row from making the row
        # appear fully skippable.
        for inspected_candidate in final_candidates.candidates:
            source = inspected_candidate.candidate.source
            final_source_keys.append(source)
            requests = code_test.execution_requests(
                CodeArtifact(source=source),
                {code_test.settings.task_key: task_artifact},
            )
            candidate_request_keys = tuple(
                _execution_request_key(request) for request in requests
            )
            if not candidate_request_keys:
                raise RuntimeError(
                    "final candidate produced no execution requests"
                )
            candidate_request_hits = tuple(
                execution_cache.get(key) is not None
                for key in candidate_request_keys
            )
            if all(candidate_request_hits):
                candidate_hits += 1
            row_missed_request_keys.update(
                key
                for key, hit in zip(
                    candidate_request_keys,
                    candidate_request_hits,
                    strict=True,
                )
                if not hit
            )
            row_request_keys.extend(candidate_request_keys)

        row_results.append(
            _RowCacheResult(
                failure_code=None,
                candidate_count=len(final_candidates.candidates),
                candidate_hits=candidate_hits,
            )
        )
        execution_request_keys.extend(row_request_keys)
        for key in row_missed_request_keys:
            execution_cache.put(key, placeholder_outcome)

    return _CandidateMeasurements(
        raw_source_keys=tuple(raw_source_keys),
        final_source_keys=tuple(final_source_keys),
        execution_request_keys=tuple(execution_request_keys),
        row_cache=_summarize_row_cache(row_results),
    )


def _benchmark_task(
    task: HumanEvalTask,
    task_rows: pl.DataFrame,
) -> _TaskBenchmark:
    decoder_outputs: list[str] = task_rows.get_column(
        "decoder_output"
    ).to_list()

    runner = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    preprocessing_started = perf_counter()
    traces = [runner.run(TextArtifact(text=text)) for text in decoder_outputs]
    preprocessing_seconds = perf_counter() - preprocessing_started
    test_cache_started = perf_counter()
    candidates = _candidate_measurements(traces, task)
    test_cache_seconds = perf_counter() - test_cache_started
    return _TaskBenchmark(
        task_id=task.task_id,
        rows=len(decoder_outputs),
        distinct_outputs=len(set(decoder_outputs)),
        preprocessing_seconds=preprocessing_seconds,
        test_cache_seconds=test_cache_seconds,
        candidates=candidates,
    )


def _aggregate_combined_factor(results: Sequence[_TaskBenchmark]) -> float:
    preprocessing_seconds = sum(
        result.preprocessing_seconds for result in results
    )
    combined_seconds = sum(result.combined_seconds for result in results)
    return combined_seconds / preprocessing_seconds


def _aggregate_preprocessing_success_rate(
    results: Sequence[_TaskBenchmark],
) -> float:
    successful_rows = sum(
        result.row_cache.successful_rows for result in results
    )
    rows = sum(result.rows for result in results)
    return successful_rows / rows


def _aggregate_testable_candidate_hit_rate(
    results: Sequence[_TaskBenchmark],
) -> float:
    candidate_hits = sum(result.row_cache.candidate_hits for result in results)
    candidate_count = sum(
        result.row_cache.candidate_count for result in results
    )
    return candidate_hits / candidate_count if candidate_count else 0.0


def _aggregate_overall_full_skip_rate(
    results: Sequence[_TaskBenchmark],
) -> float:
    fully_cached_rows = sum(
        result.row_cache.fully_cached_rows for result in results
    )
    rows = sum(result.rows for result in results)
    return fully_cached_rows / rows if rows else 0.0


def _aggregate_conditional_full_skip_rate(
    results: Sequence[_TaskBenchmark],
) -> float:
    fully_cached_rows = sum(
        result.row_cache.fully_cached_rows for result in results
    )
    successful_rows = sum(
        result.row_cache.successful_rows for result in results
    )
    return fully_cached_rows / successful_rows if successful_rows else 0.0


def _combined_row_cache(
    results: Sequence[_TaskBenchmark],
) -> _RowCacheSummary:
    summaries = tuple(result.row_cache for result in results)
    failure_codes: Counter[str] = Counter()
    for summary in summaries:
        failure_codes.update(dict(summary.failure_codes))
    return _RowCacheSummary(
        rows=sum(summary.rows for summary in summaries),
        failed_rows=sum(summary.failed_rows for summary in summaries),
        successful_rows=sum(summary.successful_rows for summary in summaries),
        zero_hit_rows=sum(summary.zero_hit_rows for summary in summaries),
        partial_hit_rows=sum(
            summary.partial_hit_rows for summary in summaries
        ),
        fully_cached_rows=sum(
            summary.fully_cached_rows for summary in summaries
        ),
        candidate_count=sum(summary.candidate_count for summary in summaries),
        candidate_hits=sum(summary.candidate_hits for summary in summaries),
        uncached_candidates=tuple(
            value
            for summary in summaries
            for value in summary.uncached_candidates
        ),
        failure_codes=tuple(sorted(failure_codes.items())),
    )


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


def _print_reuse_stats(label: str, stats: _ReuseStats) -> None:
    print(
        f"{label}: {stats.hit_rate:.2%} "
        f"({stats.hits:,}/{stats.occurrences:,} hits; "
        f"{stats.unique_keys:,} unique)"
    )


def _print_row_cache_summary(summary: _RowCacheSummary) -> None:
    rows = summary.rows
    successful_rows = summary.successful_rows
    print(
        "Testable candidate cache hit rate: "
        f"{summary.candidate_hit_rate:.2%} "
        f"({summary.candidate_hits:,}/{summary.candidate_count:,})"
    )
    print(
        "Overall full-test skip rate: "
        f"{summary.overall_full_skip_rate:.2%} "
        f"({summary.fully_cached_rows:,}/{rows:,} nonblank rows)"
    )
    print(
        "Conditional full-test skip rate: "
        f"{summary.conditional_full_skip_rate:.2%} "
        f"({summary.fully_cached_rows:,}/{successful_rows:,} "
        "successful rows)"
    )
    print("Row outcomes:")
    for label, count in (
        ("preprocessing failed", summary.failed_rows),
        ("successful, zero cache hits", summary.zero_hit_rows),
        ("successful, partial cache hits", summary.partial_hit_rows),
        ("successful, fully cached", summary.fully_cached_rows),
    ):
        rate = count / rows if rows else 0.0
        print(f"  {label}: {count:,} ({rate:.2%})")

    distribution = summary.uncached_candidate_distribution
    print(
        "Uncached candidates per successful row: "
        f"total={distribution.total:,}, mean={distribution.mean:.2f}, "
        f"median={distribution.median:.2f}, p95={distribution.p95:,}"
    )
    print("Preprocessing failure codes:")
    if not summary.failure_codes:
        print("  none")
    else:
        for failure_code, count in summary.failure_codes:
            rate = count / summary.failed_rows
            print(f"  {failure_code}: {count:,} ({rate:.2%} of failures)")


def _print_single_task(result: _TaskBenchmark) -> None:
    print(f"Filtered rows: {result.rows:,}")
    print(f"Distinct decoder outputs: {result.distinct_outputs:,}")
    print(
        f"Preprocessing success: {result.success_rate:.2%} "
        f"({result.row_cache.successful_rows:,}/{result.rows:,} "
        "nonblank samples)"
    )
    print(
        f"Full preprocessing: {result.preprocessing_seconds:.6f} seconds "
        f"({result.rows / result.preprocessing_seconds:,.2f} traces/second)"
    )
    print(
        "Test-cache planning and lookup: "
        f"{result.test_cache_seconds:.6f} seconds "
        f"(adds {result.test_cache_added_fraction:.2%} over preprocessing)"
    )
    print(
        "Preprocessing plus test-cache lookup: "
        f"{result.combined_seconds:.6f} seconds "
        f"({result.combined_factor:.2f}x preprocessing-only)"
    )
    print()
    print("Test-cache results with an initially empty in-memory cache:")
    _print_row_cache_summary(result.row_cache)
    print()
    print("Secondary source and execution-request reuse:")
    _print_reuse_stats(
        "Raw extracted-source hit rate",
        result.raw_candidate_reuse,
    )
    _print_reuse_stats(
        "Final postprocessed source-only hit rate",
        result.final_candidate_reuse,
    )
    _print_reuse_stats(
        "Execution-request hit rate",
        result.execution_request_reuse,
    )
    print("No preprocessing trace cache or candidate execution was used.")


def _print_task_sample(
    results: Sequence[_TaskBenchmark],
    *,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
) -> None:
    print(
        f"{'task':18} {'rows':>8} {'success':>8} {'success rate':>13} "
        f"{'distinct':>9} {'preprocess':>11} {'test cache':>11} "
        f"{'combined':>11} {'factor':>8}"
    )
    for result in results:
        print(
            f"{result.task_id:18} {result.rows:8,d} "
            f"{result.row_cache.successful_rows:8,d} "
            f"{result.success_rate:12.2%} "
            f"{result.distinct_outputs:9,d} "
            f"{result.preprocessing_seconds:10.3f}s "
            f"{result.test_cache_seconds:10.3f}s "
            f"{result.combined_seconds:10.3f}s "
            f"{result.combined_factor:7.2f}x"
        )

    print()
    print(
        f"{'task':18} {'candidates':>11} {'candidate hit rate':>18} "
        f"{'failed':>8} {'zero':>8} {'partial':>8} {'full':>8} "
        f"{'full skip':>10}"
    )
    for result in results:
        summary = result.row_cache
        print(
            f"{result.task_id:18} {summary.candidate_count:11,d} "
            f"{summary.candidate_hit_rate:14.2%} "
            f"{summary.failed_rows:8,d} {summary.zero_hit_rows:8,d} "
            f"{summary.partial_hit_rows:8,d} "
            f"{summary.fully_cached_rows:8,d} "
            f"{summary.overall_full_skip_rate:9.2%}"
        )

    print()
    _print_row_cache_summary(_combined_row_cache(results))
    print()
    print("Secondary aggregate source and execution-request reuse:")
    _print_reuse_stats(
        "Aggregate raw extracted-source hit rate",
        _reuse_stats(
            key
            for result in results
            for key in result.candidates.raw_source_keys
        ),
    )
    _print_reuse_stats(
        "Aggregate final postprocessed source-only hit rate",
        _reuse_stats(
            key
            for result in results
            for key in result.candidates.final_source_keys
        ),
    )
    _print_reuse_stats(
        "Aggregate execution-request hit rate",
        _reuse_stats(
            key
            for result in results
            for key in result.candidates.execution_request_keys
        ),
    )

    combined_factor_interval = bootstrap_confidence_interval(
        results,
        _aggregate_combined_factor,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    success_rate_interval = bootstrap_confidence_interval(
        results,
        _aggregate_preprocessing_success_rate,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    testable_candidate_interval = bootstrap_confidence_interval(
        results,
        _aggregate_testable_candidate_hit_rate,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    overall_full_skip_interval = bootstrap_confidence_interval(
        results,
        _aggregate_overall_full_skip_rate,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    conditional_full_skip_interval = bootstrap_confidence_interval(
        results,
        _aggregate_conditional_full_skip_rate,
        confidence_level=confidence_level,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    print()
    _print_interval(
        "Aggregate preprocessing success rate",
        success_rate_interval,
        percentage=True,
    )
    _print_interval(
        "Aggregate preprocessing-plus-test-cache factor",
        combined_factor_interval,
        percentage=False,
    )
    _print_interval(
        "Aggregate testable candidate cache hit rate",
        testable_candidate_interval,
        percentage=True,
    )
    _print_interval(
        "Aggregate overall full-test skip rate",
        overall_full_skip_interval,
        percentage=True,
    )
    _print_interval(
        "Aggregate conditional full-test skip rate",
        conditional_full_skip_interval,
        percentage=True,
    )
    print(
        f"Bootstrap unit: task; resamples: {bootstrap_resamples:,}; "
        f"seed: {seed}"
    )
    print(
        "Test-cache lookup used exact execution-request keys without "
        "executing candidate code."
    )


def _load_selected_tasks(
    snapshot_path: Path,
    selected_task_ids: Sequence[str],
) -> dict[str, HumanEvalTask]:
    selected = set(selected_task_ids)
    rows = [
        row
        for row in load_humaneval_rows(snapshot_path=snapshot_path)
        if str(row["task_id"]) in selected
    ]
    tasks = {task.task_id: task for task in parse_humaneval_dataset(rows)}
    missing = selected.difference(tasks)
    if missing:
        raise ValueError(
            "HumanEval snapshot is missing selected tasks: "
            + ", ".join(sorted(missing))
        )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exhaustive preprocessing followed by in-memory "
            "test-result cache lookup without executing candidates."
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
    parser.add_argument(
        "--humaneval-snapshot",
        type=_parquet_path,
        default=_DEFAULT_HUMANEVAL_SNAPSHOT,
        help=(
            "pinned HumanEval raw-row snapshot used only to construct "
            "test-safe cache keys "
            f"(default: {_DEFAULT_HUMANEVAL_SNAPSHOT})"
        ),
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

    task_load_started = perf_counter()
    try:
        tasks_by_id = _load_selected_tasks(
            arguments.humaneval_snapshot,
            selected_task_ids,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    task_load_seconds = perf_counter() - task_load_started

    results: list[_TaskBenchmark] = []
    for index, task_id in enumerate(selected_task_ids, start=1):
        task_rows = selected_rows.filter(pl.col("task_id") == task_id)
        print(
            f"Benchmarking task {index}/{len(selected_task_ids)}: "
            f"{task_id} ({task_rows.height:,} rows)",
            flush=True,
        )
        results.append(_benchmark_task(tasks_by_id[task_id], task_rows))

    print(f"Parquet: {arguments.parquet}")
    print(
        f"Loaded corpus: {corpus.height:,} rows x {corpus.width} columns "
        f"in {load_seconds:.6f} seconds"
    )
    print(
        f"Selected and filtered: {selected_rows.height:,} rows in "
        f"{filter_seconds:.6f} seconds"
    )
    print(
        f"Loaded {len(tasks_by_id):,} selected HumanEval task payloads in "
        f"{task_load_seconds:.6f} seconds"
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
