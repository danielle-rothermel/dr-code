from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
BASELINE_ROOT: Final = Path(__file__).resolve().parent / "baseline"
GENERATION_CORPUS_BUNDLE: Final = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "generation_corpus" / "human_eval"
)
GENERATION_CORPUS_BUNDLE_ENV: Final = "DR_CODE_GENERATION_CORPUS_BUNDLE"
EXPECTED_MANIFEST_SHA256_ENV: Final = "DR_CODE_EXPECTED_MANIFEST_SHA256"
RUN_DIRECTORY_ENV: Final = "DR_CODE_TASK_DIFFICULTY_RUN_DIR"
DEFAULT_RUN_DIRECTORY: Final = (
    Path.home()
    / "drotherm"
    / "data"
    / ".codex"
    / "dr-code"
    / "task-difficulty-directional"
    / "runs"
    / "default"
)
HUMANEVAL_SNAPSHOT: Final = (
    REPOSITORY_ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
)
SAMPLING_SEED: Final = 42
EVAL_WORKERS: Final = 16
EVAL_TIMEOUT_SECONDS: Final = 120.0
PREPROCESS_TIMEOUT_SECONDS_ENV: Final = "DR_CODE_PREPROCESS_TIMEOUT_SECONDS"
PREPROCESS_TIMEOUT_SECONDS: Final = 600.0
"""Per-item preprocessing wall-time watchdog for this workflow.

Deliberately far above any healthy input's cost: it exists to break a wedged
worker, not to bound normal work, and should never fire on a healthy corpus.
"""
SAMPLE_TASKS_PER_GROUP_ENV: Final = "DR_CODE_SAMPLE_TASKS_PER_GROUP"
SAMPLE_TASKS_PER_GROUP: Final = 40
"""Tasks retained per populated (generation_mode, budget_mode, model_key) group.

This baseline is a PR-versus-PR regression probe, so it needs constant group
membership rather than a balanced design. Runtime is bounded by seeded
deterministic task subsetting instead of by narrowing which cells are observed.
"""


def run_directory_path() -> Path:
    override = os.environ.get(RUN_DIRECTORY_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_RUN_DIRECTORY


def generation_corpus_bundle_path() -> Path:
    override = os.environ.get(GENERATION_CORPUS_BUNDLE_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return GENERATION_CORPUS_BUNDLE


def expected_manifest_sha256() -> str | None:
    override = os.environ.get(EXPECTED_MANIFEST_SHA256_ENV)
    if override:
        return override
    return None


def baseline_directory(name: str) -> Path:
    return BASELINE_ROOT / name


def parse_preprocess_timeout_seconds(value: str) -> float | None:
    """Return a positive wall-time budget, or `None` for an explicit opt-out.

    `0` and `none` both mean unbudgeted, matching the library default.
    """

    if value.strip().lower() in {"none", "0"}:
        return None
    try:
        timeout_seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a number, 0, or 'none'"
        ) from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise argparse.ArgumentTypeError(
            "must be finite and positive, or 0/'none' for unbudgeted"
        )
    return timeout_seconds


def preprocess_timeout_seconds() -> float | None:
    """Return this workflow's per-item preprocessing wall-time budget."""

    override = os.environ.get(PREPROCESS_TIMEOUT_SECONDS_ENV)
    if override:
        return parse_preprocess_timeout_seconds(override)
    return PREPROCESS_TIMEOUT_SECONDS


def parse_sample_tasks_per_group(value: str) -> int | None:
    """Return a positive task budget per group, or `None` for every task.

    `0` and `all` both mean the full task grid.
    """

    if value.strip().lower() in {"all", "0"}:
        return None
    try:
        tasks_per_group = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be an integer, 0, or 'all'"
        ) from exc
    if tasks_per_group < 1:
        raise argparse.ArgumentTypeError(
            "must be positive, or 0/'all' for every task"
        )
    return tasks_per_group


def sample_tasks_per_group() -> int | None:
    """Return the tasks retained per populated sampling group."""

    override = os.environ.get(SAMPLE_TASKS_PER_GROUP_ENV)
    if override:
        return parse_sample_tasks_per_group(override)
    return SAMPLE_TASKS_PER_GROUP


_RUN_DIRECTORY = run_directory_path()
RUN_DIRECTORY: Final = _RUN_DIRECTORY
EVAL_RUN_ROOT: Final = _RUN_DIRECTORY / "explicit-runtime"
ELIGIBLE_CORPUS: Final = _RUN_DIRECTORY / "eligible_generations.parquet"
PREPROCESSING_SUMMARY: Final = _RUN_DIRECTORY / "preprocessing_summary.parquet"
SELECTED_SAMPLE: Final = _RUN_DIRECTORY / "selected_sample.parquet"
SAMPLING_COVERAGE: Final = _RUN_DIRECTORY / "sampling_coverage.parquet"
PREPROCESS_LOG: Final = _RUN_DIRECTORY / "01_preprocess.log"
SAMPLING_LOG: Final = _RUN_DIRECTORY / "02_sample.log"


@dataclass(frozen=True, slots=True)
class EvalSettings:
    worker_count: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class EvalPaths:
    root: Path
    bundle_root: Path
    execution_cache: Path
    evaluation_object_store: Path
    run_manifest: Path
    execution_records: Path
    candidate_results: Path
    generation_results: Path
    task_setting_results: Path
    task_results: Path
    evaluation_log: Path
    summary_log: Path


def _positive_worker_count(value: str) -> int:
    try:
        worker_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if worker_count < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return worker_count


def _positive_timeout_seconds(value: str) -> float:
    try:
        timeout_seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return timeout_seconds


def parse_eval_args(
    description: str | None,
    argv: Sequence[str] | None = None,
) -> EvalSettings:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--workers",
        type=_positive_worker_count,
        default=EVAL_WORKERS,
        help=(
            "concurrent workers for preprocessing (stage 1) and candidate "
            f"execution (stage 3 global pool capacity; default: {EVAL_WORKERS})"
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_timeout_seconds,
        default=EVAL_TIMEOUT_SECONDS,
        help=(
            "maximum wall time for one candidate batch "
            f"(default: {EVAL_TIMEOUT_SECONDS:g})"
        ),
    )
    arguments = parser.parse_args(argv)
    return EvalSettings(
        worker_count=arguments.workers,
        timeout_seconds=arguments.timeout_seconds,
    )


def eval_paths(settings: EvalSettings) -> EvalPaths:
    timeout_label = format(settings.timeout_seconds, ".17g").replace(".", "p")
    root = EVAL_RUN_ROOT / (
        f"workers-{settings.worker_count}_timeout-{timeout_label}"
    )
    return EvalPaths(
        root=root,
        bundle_root=root / "evaluation_bundles",
        execution_cache=root / "execution_cache.sqlite3",
        evaluation_object_store=root / "evaluation_object_store.sqlite3",
        run_manifest=root / "run_manifest.json",
        execution_records=root / "execution_records",
        candidate_results=root / "candidate_results.parquet",
        generation_results=root / "generation_results.parquet",
        task_setting_results=root / "task_setting_results.parquet",
        task_results=root / "task_results.parquet",
        evaluation_log=root / "03_evaluate.log",
        summary_log=root / "04_summarize.log",
    )


def prepare_run_directory() -> None:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
