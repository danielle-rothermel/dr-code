from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[3]
GENERATION_CORPUS: Final = (
    Path.home()
    / "drotherm"
    / "repos"
    / "gen-viewer"
    / "data"
    / "generation-corpus.parquet"
)
HUMANEVAL_SNAPSHOT: Final = (
    REPOSITORY_ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
)
RUN_DIRECTORY: Final = (
    Path.home()
    / "drotherm"
    / "data"
    / ".codex"
    / "dr-code"
    / "2026-08-06"
    / "task-difficulty-directional"
)
EVALUATION_RUN_ROOT: Final = RUN_DIRECTORY / "explicit-runtime"

ELIGIBLE_CORPUS: Final = RUN_DIRECTORY / "eligible_generations.parquet"
PREPROCESSING_SUMMARY: Final = RUN_DIRECTORY / "preprocessing_summary.parquet"
PREPROCESSING_CACHE: Final = RUN_DIRECTORY / "preprocessing_cache.sqlite3"
SELECTED_SAMPLE: Final = RUN_DIRECTORY / "selected_sample.parquet"
SAMPLING_COVERAGE: Final = RUN_DIRECTORY / "sampling_coverage.parquet"
SOURCE_KIND: Final = "legacy_dbos_generation_attempt"
SAMPLING_SEED: Final = 42
EVALUATION_WORKERS: Final = 16
EVALUATION_TIMEOUT_SECONDS: Final = 120.0
SETTINGS: Final = (
    ("direct", "no_budget"),
    ("enc_dec", "no_budget"),
    ("enc_dec", "budget"),
)
MODEL_ROSTER: Final = (
    "deepseek/deepseek-v3.1-terminus",
    "openai/gpt-5-nano",
    "qwen/qwen3-coder-flash",
)

PREPROCESS_LOG: Final = RUN_DIRECTORY / "01_preprocess.log"
SAMPLING_LOG: Final = RUN_DIRECTORY / "02_sample.log"


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    worker_count: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationPaths:
    root: Path
    parts: Path
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


def parse_evaluation_args(
    description: str | None,
    argv: Sequence[str] | None = None,
) -> EvaluationSettings:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--workers",
        type=_positive_worker_count,
        default=EVALUATION_WORKERS,
        help=f"concurrent task workers (default: {EVALUATION_WORKERS})",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_timeout_seconds,
        default=EVALUATION_TIMEOUT_SECONDS,
        help=(
            "maximum wall time for one candidate batch "
            f"(default: {EVALUATION_TIMEOUT_SECONDS:g})"
        ),
    )
    arguments = parser.parse_args(argv)
    return EvaluationSettings(
        worker_count=arguments.workers,
        timeout_seconds=arguments.timeout_seconds,
    )


def evaluation_paths(settings: EvaluationSettings) -> EvaluationPaths:
    timeout_label = format(settings.timeout_seconds, ".17g").replace(".", "p")
    root = EVALUATION_RUN_ROOT / (
        f"workers-{settings.worker_count}_timeout-{timeout_label}"
    )
    return EvaluationPaths(
        root=root,
        parts=root / "evaluation_parts",
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
