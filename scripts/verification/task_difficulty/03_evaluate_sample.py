#!/usr/bin/env python3

"""Evaluate the selected historical candidates on a disposable worker."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import polars as pl
from dr_exec import Executor

from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.metric_operator import CodeTestSettings
from dr_code.humaneval.sampling import load_humaneval_rows
from dr_code.metrics import (
    MeasuredRecord,
    MetricName,
    MetricQuestion,
    MetricsDefinition,
    OperatorFailureRecord,
    extract_metrics_batch,
)
from dr_code.trace import CodeArtifact, JsonArtifact, external_trace

from workflow_settings import (
    EVALUATION_LOG,
    EVALUATION_PARTS,
    HUMANEVAL_SNAPSHOT,
    SELECTED_SAMPLE,
    prepare_run_directory,
)

_DERIVED_TASK_FIELDS = frozenset(
    {"parsed", "parsed_tests", *HumanEvalTask.model_computed_fields}
)
_METRICS = MetricsDefinition(
    definition_id="directional-humaneval-task-difficulty",
    version="0",
    questions=(
        MetricQuestion(
            metric=MetricName.CODE_TEST,
            on="output",
            settings=CodeTestSettings(),
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


def _metric_values(record: object) -> dict[str, object]:
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
            "metric_status": "operator_failure",
            "candidate_passed": None,
            "failure_type": record.failure.failure_type,
            "failure_message": record.failure.failure_message,
        }
    return {
        "metric_status": "not_applicable",
        "candidate_passed": None,
        "failure_type": None,
        "failure_message": None,
    }


def evaluate_task_rows(
    task_rows: pl.DataFrame,
    task: HumanEvalTask,
    *,
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
        _METRICS,
        traces,
        executor=executor,
    )
    records: list[dict[str, object]] = []
    for identity, batch in zip(identities, record_batches, strict=True):
        if len(batch) != 1:
            raise RuntimeError("expected exactly one code-test metric record")
        records.append({**identity, **_metric_values(batch[0])})
    return pl.DataFrame(records, infer_schema_length=None)


def _part_path(task_id: str) -> Path:
    return EVALUATION_PARTS / f"{task_id.replace('/', '_')}.parquet"


def _validate_existing_part(path: Path, task_rows: pl.DataFrame) -> None:
    existing = pl.read_parquet(path)
    expected_rows = int(task_rows.get_column("candidate_count").sum())
    expected_samples = set(task_rows.get_column("sample_id").to_list())
    actual_samples = set(existing.get_column("sample_id").to_list())
    if existing.height != expected_rows or actual_samples != expected_samples:
        raise RuntimeError(
            f"existing evaluation part does not match current sample: {path}"
        )


def main() -> int:
    if os.environ.get("DR_CODE_DISPOSABLE_WORKER") != "1":
        raise SystemExit(
            "refusing to execute historical model output: run this stage "
            "only on a disposable worker and set DR_CODE_DISPOSABLE_WORKER=1"
        )

    prepare_run_directory()
    EVALUATION_PARTS.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(EVALUATION_LOG)
    selected = pl.read_parquet(SELECTED_SAMPLE)
    task_ids: list[str] = (
        selected.get_column("task_id").unique().sort().to_list()
    )
    tasks = _load_tasks(HUMANEVAL_SNAPSHOT, task_ids)
    logger.info(
        "Loaded %,d selected generations across %,d tasks",
        selected.height,
        len(task_ids),
    )

    for index, task_id in enumerate(task_ids, start=1):
        task_rows = selected.filter(pl.col("task_id") == task_id)
        part_path = _part_path(task_id)
        if part_path.exists():
            _validate_existing_part(part_path, task_rows)
            logger.info(
                "Skipping completed task %d/%d: %s",
                index,
                len(task_ids),
                task_id,
            )
            continue

        started = perf_counter()
        logger.info(
            "Evaluating task %d/%d: %s (%d generations, %d candidates)",
            index,
            len(task_ids),
            task_id,
            task_rows.height,
            task_rows.get_column("candidate_count").sum(),
        )
        results = evaluate_task_rows(task_rows, tasks[task_id])
        temporary_path = part_path.with_suffix(".tmp.parquet")
        results.write_parquet(temporary_path)
        temporary_path.replace(part_path)
        logger.info(
            "Completed %s in %.1f seconds",
            task_id,
            perf_counter() - started,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
