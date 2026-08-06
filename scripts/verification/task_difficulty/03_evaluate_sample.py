#!/usr/bin/env python3

"""Evaluate the selected historical candidates on a disposable worker."""

from __future__ import annotations

import json
import logging
import os
import resource
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

import polars as pl
from dr_exec import Executor, ProcessExecutor

from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.metric_operator import CodeTestSettings
from dr_code.humaneval.sampling import load_humaneval_rows
from dr_code.core.execution.executor import (
    host_process_executor,
    run_python_source,
)
from dr_code.metrics import (
    MeasuredRecord,
    MetricName,
    MetricQuestion,
    MetricRecord,
    MetricsDefinition,
    OperatorFailureRecord,
    extract_metrics_batch,
)
from dr_code.trace import CodeArtifact, JsonArtifact, external_trace

from workflow_settings import (
    EVALUATION_TIMEOUT_SECONDS,
    EvaluationPaths,
    EvaluationSettings,
    HUMANEVAL_SNAPSHOT,
    SELECTED_SAMPLE,
    evaluation_paths,
    parse_evaluation_args,
    prepare_run_directory,
)

_DERIVED_TASK_FIELDS = frozenset(
    {"parsed", "parsed_tests", *HumanEvalTask.model_computed_fields}
)
_RUNTIME_ENVIRONMENT_VARIABLE = "DR_CODE_EVALUATION_PYTHON"
_MINIMUM_OPEN_FILE_LIMIT = 4096
_OPEN_FILES_PER_WORKER = 64
_RUNTIME_PROBE_SOURCE = """\
import json
import platform
import sys

def dr_exec_main(request, emit):
    import numpy
    print(json.dumps({
        "implementation": platform.python_implementation(),
        "numpy_version": numpy.__version__,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
    }, sort_keys=True, separators=(",", ":")))
"""


@dataclass(frozen=True, slots=True)
class _TaskJob:
    task_id: str
    task_rows: pl.DataFrame
    task: HumanEvalTask
    runtime_executable: Path
    runtime_identity: str
    evaluation_settings: EvaluationSettings
    evaluation_paths: EvaluationPaths


@dataclass(frozen=True, slots=True)
class _TaskCompletion:
    task_id: str
    generation_count: int
    candidate_count: int
    elapsed_seconds: float


def _metrics_definition(timeout_seconds: float) -> MetricsDefinition:
    return MetricsDefinition(
        definition_id="directional-humaneval-task-difficulty",
        version="0",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="output",
                settings=CodeTestSettings(timeout_seconds=timeout_seconds),
            ),
        ),
    )


def _configure_logging(path: Path) -> logging.Logger:
    logger = logging.getLogger("task_difficulty.evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(path, mode="a", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _ensure_open_file_limit(
    worker_count: int,
    logger: logging.Logger,
) -> int:
    required = max(
        _MINIMUM_OPEN_FILE_LIMIT,
        worker_count * _OPEN_FILES_PER_WORKER,
    )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= required:
        logger.info(
            "Open-file limit is sufficient: soft=%d required=%d",
            soft,
            required,
        )
        return soft
    if hard != resource.RLIM_INFINITY and hard < required:
        raise SystemExit(
            f"open-file soft limit {soft} is below the required {required}, "
            f"and the hard limit {hard} prevents raising it; run "
            f"`ulimit -n {required}` in the launching shell"
        )
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"could not raise the open-file soft limit from {soft} to "
            f"{required}; run `ulimit -n {required}` in the launching shell: "
            f"{exc}"
        ) from exc
    effective, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if effective < required:
        raise SystemExit(
            f"open-file soft limit remained {effective}, below the required "
            f"{required}; run `ulimit -n {required}` in the launching shell"
        )
    logger.info(
        "Raised open-file soft limit from %d to %d for %d workers",
        soft,
        effective,
        worker_count,
    )
    return effective


def _load_tasks(
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
            "HumanEval snapshot is missing tasks: "
            + ", ".join(sorted(missing))
        )
    return tasks


def _metric_values(record: MetricRecord) -> dict[str, object]:
    metric_identity = record.identity
    identity_values = {
        "metric_schema_version": record.schema_version,
        "metric_name": str(metric_identity.question.metric),
        "metric_version": metric_identity.metric_version,
        "metrics_definition_id": (
            metric_identity.metrics_definition.definition_id
        ),
        "metrics_definition_version": (
            metric_identity.metrics_definition.version
        ),
    }
    if isinstance(record, MeasuredRecord):
        facts = {fact.name: fact.value for fact in record.facts}
        total_cases = facts["total_cases"]
        passed_count = facts["passed_count"]
        coverage_complete = facts["coverage_complete"]
        if (
            isinstance(total_cases, bool)
            or not isinstance(total_cases, int)
            or isinstance(passed_count, bool)
            or not isinstance(passed_count, int)
            or not isinstance(coverage_complete, bool)
        ):
            raise TypeError("code-test metric returned invalid fact types")
        return {
            **identity_values,
            "metric_status": "measured",
            **facts,
            "candidate_passed": (
                coverage_complete and passed_count == total_cases
            ),
            "failure_type": None,
            "failure_message": None,
        }
    if isinstance(record, OperatorFailureRecord):
        return {
            **identity_values,
            "metric_status": "operator_failure",
            "candidate_passed": None,
            "failure_type": record.failure.failure_type,
            "failure_message": record.failure.failure_message,
        }
    return {
        **identity_values,
        "metric_status": "not_applicable",
        "candidate_passed": None,
        "failure_type": None,
        "failure_message": None,
    }


def evaluate_task_rows(
    task_rows: pl.DataFrame,
    task: HumanEvalTask,
    *,
    timeout_seconds: float = EVALUATION_TIMEOUT_SECONDS,
    executor: Executor | None = None,
) -> pl.DataFrame:
    task_artifact = JsonArtifact(
        payload=task.model_dump(
            mode="json",
            exclude=set(_DERIVED_TASK_FIELDS),
        )
    )
    identities: list[dict[str, object]] = []
    traces = []
    for row in task_rows.iter_rows(named=True):
        candidates = row["code_candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f"selected sample {row['sample_id']!r} has no candidates"
            )
        for candidate_index, source in enumerate(candidates):
            if not isinstance(source, str):
                raise TypeError("candidate source must be a string")
            code = CodeArtifact(source=source)
            traces.append(
                external_trace(
                    {
                        "input": code,
                        "output": code,
                        "task": task_artifact,
                    }
                )
            )
            identities.append(
                {
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "generation_mode": row["generation_mode"],
                    "budget_mode": row["budget_mode"],
                    "model_key": row["model_key"],
                    "candidate_index": candidate_index,
                    "candidate_source": source,
                }
            )

    record_batches = extract_metrics_batch(
        _metrics_definition(timeout_seconds),
        traces,
        executor=executor,
    )
    records: list[dict[str, object]] = []
    for identity, batch in zip(identities, record_batches, strict=True):
        if len(batch) != 1:
            raise RuntimeError("expected exactly one code-test metric record")
        records.append({**identity, **_metric_values(batch[0])})
    return pl.DataFrame(records, infer_schema_length=None)


def _part_path(parts_directory: Path, task_id: str) -> Path:
    return parts_directory / f"{task_id.replace('/', '_')}.parquet"


def _validate_existing_part(
    path: Path,
    task_rows: pl.DataFrame,
    runtime_identity: str,
    evaluation_settings: EvaluationSettings,
) -> None:
    existing = pl.read_parquet(path)
    expected_rows = int(task_rows.get_column("candidate_count").sum())
    expected_candidates = sorted(
        (str(row["sample_id"]), index, source)
        for row in task_rows.iter_rows(named=True)
        for index, source in enumerate(row["code_candidates"])
    )
    actual_candidates = sorted(
        existing.select(
            ["sample_id", "candidate_index", "candidate_source"]
        ).iter_rows()
    )
    identities = (
        existing.get_column("runtime_identity").unique().to_list()
        if "runtime_identity" in existing.columns
        else []
    )
    settings = (
        existing.select(
            ["evaluation_worker_count", "evaluation_timeout_seconds"]
        )
        .unique()
        .to_dicts()
        if {
            "evaluation_worker_count",
            "evaluation_timeout_seconds",
        }.issubset(existing.columns)
        else []
    )
    expected_settings = [
        {
            "evaluation_worker_count": evaluation_settings.worker_count,
            "evaluation_timeout_seconds": (
                evaluation_settings.timeout_seconds
            ),
        }
    ]
    if (
        existing.height != expected_rows
        or actual_candidates != expected_candidates
        or identities != [runtime_identity]
        or settings != expected_settings
    ):
        raise RuntimeError(
            "existing evaluation part does not match the current sample, "
            f"runtime, and evaluation settings: {path}"
        )


def _require_measured_results(results: pl.DataFrame, context: str) -> None:
    failures = results.filter(pl.col("metric_status") != "measured")
    if failures.is_empty():
        return
    examples = (
        failures.select("failure_type", "failure_message")
        .unique()
        .head(3)
        .to_dicts()
    )
    raise RuntimeError(
        f"{context} produced {failures.height} harness/operator failures: "
        f"{examples}"
    )


def _runtime_executable_from_environment() -> Path:
    value = os.environ.get(_RUNTIME_ENVIRONMENT_VARIABLE)
    if not value:
        raise SystemExit(
            f"{_RUNTIME_ENVIRONMENT_VARIABLE} must name a copied Python "
            "executable with dr-code's runtime dependencies installed"
        )
    executable = Path(value).expanduser().absolute()
    if not executable.is_file():
        raise SystemExit(f"evaluation Python is not a file: {executable}")
    if executable.is_symlink():
        raise SystemExit(
            "evaluation Python must be a copied executable, not a symlink: "
            f"{executable}"
        )
    return executable


def _runtime_identity(executor: ProcessExecutor) -> str:
    completed = run_python_source(
        executor,
        source=_RUNTIME_PROBE_SOURCE,
        input_json="{}",
        timeout_seconds=10.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "evaluation runtime dependency probe failed: "
            + completed.stderr.strip()
        )
    try:
        package_identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "evaluation runtime dependency probe returned invalid JSON"
        ) from exc
    identity = {
        "runtime": executor.runtime.describe().id_doc.to_json_dict(),
        "packages": package_identity,
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _preflight_runtime(
    runtime_executable: Path,
    task: HumanEvalTask,
    evaluation_settings: EvaluationSettings,
    paths: EvaluationPaths,
) -> str:
    record_directory = paths.execution_records / "preflight"
    record_directory.mkdir(parents=True, exist_ok=True)
    executor = host_process_executor(
        record_directory,
        runtime_executable=runtime_executable,
    )
    runtime_identity = _runtime_identity(executor)
    selected = pl.DataFrame(
        [
            {
                "sample_id": "runtime-preflight",
                "task_id": task.task_id,
                "generation_mode": "preflight",
                "budget_mode": "preflight",
                "model_key": "ground-truth",
                "code_candidates": [task.ground_truth_code],
                "candidate_count": 1,
            }
        ]
    )
    results = evaluate_task_rows(
        selected,
        task,
        timeout_seconds=evaluation_settings.timeout_seconds,
        executor=executor,
    )
    _require_measured_results(results, "runtime preflight")
    if results.item(0, "candidate_passed") is not True:
        raise RuntimeError(
            "runtime preflight ground-truth candidate did not pass"
        )
    return runtime_identity


def _run_task_job(job: _TaskJob) -> _TaskCompletion:
    started = perf_counter()
    record_directory = job.evaluation_paths.execution_records / (
        job.task_id.replace("/", "_")
    )
    record_directory.mkdir(parents=True, exist_ok=True)
    executor = host_process_executor(
        record_directory,
        runtime_executable=job.runtime_executable,
    )
    results = evaluate_task_rows(
        job.task_rows,
        job.task,
        timeout_seconds=job.evaluation_settings.timeout_seconds,
        executor=executor,
    )
    _require_measured_results(results, job.task_id)
    results = results.with_columns(
        pl.lit(job.runtime_identity).alias("runtime_identity"),
        pl.lit(job.evaluation_settings.worker_count).alias(
            "evaluation_worker_count"
        ),
        pl.lit(job.evaluation_settings.timeout_seconds).alias(
            "evaluation_timeout_seconds"
        ),
    )
    part_path = _part_path(job.evaluation_paths.parts, job.task_id)
    temporary_path = part_path.with_suffix(".tmp.parquet")
    results.write_parquet(temporary_path)
    temporary_path.replace(part_path)
    return _TaskCompletion(
        task_id=job.task_id,
        generation_count=job.task_rows.height,
        candidate_count=int(job.task_rows.get_column("candidate_count").sum()),
        elapsed_seconds=perf_counter() - started,
    )


def _completed_jobs(
    jobs: Sequence[_TaskJob],
    *,
    worker_count: int,
    run_job: Callable[[_TaskJob], _TaskCompletion] = _run_task_job,
) -> Iterator[tuple[_TaskJob, Future[_TaskCompletion]]]:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if not jobs:
        return
    with ThreadPoolExecutor(
        max_workers=min(worker_count, len(jobs)),
        thread_name_prefix="humaneval-task",
    ) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            yield futures[future], future


def main(argv: Sequence[str] | None = None) -> int:
    run_started = perf_counter()
    evaluation_settings = parse_evaluation_args(__doc__, argv)
    paths = evaluation_paths(evaluation_settings)
    if os.environ.get("DR_CODE_DISPOSABLE_WORKER") != "1":
        raise SystemExit(
            "refusing to execute historical model output: run this stage "
            "only on a disposable worker and set DR_CODE_DISPOSABLE_WORKER=1"
        )

    prepare_run_directory()
    paths.parts.mkdir(parents=True, exist_ok=True)
    paths.execution_records.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(paths.evaluation_log)
    logger.info(
        "Evaluation configuration: workers=%d timeout_seconds=%g "
        "run_directory=%s",
        evaluation_settings.worker_count,
        evaluation_settings.timeout_seconds,
        paths.root,
    )
    _ensure_open_file_limit(evaluation_settings.worker_count, logger)
    runtime_executable = _runtime_executable_from_environment()
    selected = pl.read_parquet(SELECTED_SAMPLE)
    task_ids: list[str] = (
        selected.get_column("task_id").unique().sort().to_list()
    )
    tasks = _load_tasks(HUMANEVAL_SNAPSHOT, task_ids)
    preflight_task_id = "HumanEval/0"
    preflight_task = _load_tasks(
        HUMANEVAL_SNAPSHOT,
        [preflight_task_id],
    )[preflight_task_id]
    runtime_identity = _preflight_runtime(
        runtime_executable,
        preflight_task,
        evaluation_settings,
        paths,
    )
    logger.info(
        "Validated evaluation runtime %s: %s",
        runtime_executable,
        runtime_identity,
    )
    logger.info(
        "Loaded %d selected generations across %d tasks",
        selected.height,
        len(task_ids),
    )

    jobs: list[_TaskJob] = []
    for task_id in task_ids:
        task_rows = selected.filter(pl.col("task_id") == task_id)
        part_path = _part_path(paths.parts, task_id)
        if part_path.exists():
            _validate_existing_part(
                part_path,
                task_rows,
                runtime_identity,
                evaluation_settings,
            )
            logger.info("Skipping completed task: %s", task_id)
            continue
        jobs.append(
            _TaskJob(
                task_id=task_id,
                task_rows=task_rows,
                task=tasks[task_id],
                runtime_executable=runtime_executable,
                runtime_identity=runtime_identity,
                evaluation_settings=evaluation_settings,
                evaluation_paths=paths,
            )
        )

    active_worker_count = min(evaluation_settings.worker_count, len(jobs))
    logger.info(
        "Starting %d pending tasks with %d concurrent workers",
        len(jobs),
        active_worker_count,
    )
    failures: list[tuple[str, str, str]] = []
    completed = 0
    evaluated_sample_count = 0
    evaluated_candidate_count = 0
    evaluation_started = perf_counter()
    for job, future in _completed_jobs(
        jobs,
        worker_count=evaluation_settings.worker_count,
    ):
        try:
            result = future.result()
        except Exception as exc:
            failures.append((job.task_id, type(exc).__name__, str(exc)))
            logger.exception("Task failed: %s", job.task_id)
            continue
        completed += 1
        evaluated_sample_count += result.generation_count
        evaluated_candidate_count += result.candidate_count
        logger.info(
            "Completed task %d/%d: %s (%d generations, %d candidates, %.1f seconds)",
            completed,
            len(jobs),
            result.task_id,
            result.generation_count,
            result.candidate_count,
            result.elapsed_seconds,
        )
    evaluation_elapsed_seconds = perf_counter() - evaluation_started
    total_elapsed_seconds = perf_counter() - run_started
    total_seconds_per_sample = (
        f"{total_elapsed_seconds / evaluated_sample_count:.6f}"
        if evaluated_sample_count
        else "n/a"
    )
    evaluation_seconds_per_sample = (
        f"{evaluation_elapsed_seconds / evaluated_sample_count:.6f}"
        if evaluated_sample_count
        else "n/a"
    )
    logger.info(
        "Evaluation timing: total_seconds=%.3f evaluation_seconds=%.3f "
        "evaluated_samples=%d evaluated_candidates=%d "
        "total_seconds_per_sample=%s evaluation_seconds_per_sample=%s "
        "configured_workers=%d active_workers=%d timeout_seconds=%g",
        total_elapsed_seconds,
        evaluation_elapsed_seconds,
        evaluated_sample_count,
        evaluated_candidate_count,
        total_seconds_per_sample,
        evaluation_seconds_per_sample,
        evaluation_settings.worker_count,
        active_worker_count,
        evaluation_settings.timeout_seconds,
    )
    if failures:
        details = "; ".join(
            f"{task_id}: {failure_type}: {message}"
            for task_id, failure_type, message in failures
        )
        raise RuntimeError(
            f"{len(failures)} task evaluations failed after other tasks "
            f"continued: {details}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
