from __future__ import annotations

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
EVALUATION_RUN_DIRECTORY: Final = RUN_DIRECTORY / "explicit-runtime"

ELIGIBLE_CORPUS: Final = RUN_DIRECTORY / "eligible_generations.parquet"
PREPROCESSING_SUMMARY: Final = RUN_DIRECTORY / "preprocessing_summary.parquet"
PREPROCESSING_CACHE: Final = RUN_DIRECTORY / "preprocessing_cache.sqlite3"
SELECTED_SAMPLE: Final = RUN_DIRECTORY / "selected_sample.parquet"
SAMPLING_COVERAGE: Final = RUN_DIRECTORY / "sampling_coverage.parquet"
EVALUATION_PARTS: Final = EVALUATION_RUN_DIRECTORY / "evaluation_parts"
EXECUTION_RECORDS: Final = EVALUATION_RUN_DIRECTORY / "execution_records"
CANDIDATE_RESULTS: Final = (
    EVALUATION_RUN_DIRECTORY / "candidate_results.parquet"
)
GENERATION_RESULTS: Final = (
    EVALUATION_RUN_DIRECTORY / "generation_results.parquet"
)
TASK_SETTING_RESULTS: Final = (
    EVALUATION_RUN_DIRECTORY / "task_setting_results.parquet"
)
TASK_RESULTS: Final = EVALUATION_RUN_DIRECTORY / "task_results.parquet"

SOURCE_KIND: Final = "legacy_dbos_generation_attempt"
SAMPLING_SEED: Final = 42
EVALUATION_WORKERS: Final = 16
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
EVALUATION_LOG: Final = EVALUATION_RUN_DIRECTORY / "03_evaluate.log"
SUMMARY_LOG: Final = EVALUATION_RUN_DIRECTORY / "04_summarize.log"


def prepare_run_directory() -> None:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    EVALUATION_RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
