#!/usr/bin/env python3

"""Copy task-difficulty run logs and summaries into a git-tracked baseline directory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from corpus_loader import load_manifest_summary, manifest_sha256
from workflow_settings import (
    ELIGIBLE_CORPUS,
    PREPROCESSING_SUMMARY,
    PREPROCESS_LOG,
    SAMPLING_COVERAGE,
    SAMPLING_LOG,
    SELECTED_SAMPLE,
    baseline_directory,
    evaluation_paths,
    generation_corpus_bundle_path,
    parse_evaluation_args,
    run_directory_path,
)


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _stage_status(run_dir: Path, paths) -> dict[str, bool]:  # noqa: ANN001
    return {
        "preprocess": (run_dir / PREPROCESS_LOG.name).is_file()
        and ELIGIBLE_CORPUS.is_file(),
        "sample": (run_dir / SAMPLING_LOG.name).is_file()
        and SELECTED_SAMPLE.is_file(),
        "evaluate": paths.evaluation_log.is_file()
        and paths.parts.is_dir()
        and any(paths.parts.glob("*.parquet")),
        "summarize": paths.summary_log.is_file()
        and paths.task_results.is_file(),
    }


def _preprocessing_summary() -> dict[str, object] | None:
    if not PREPROCESSING_SUMMARY.is_file():
        return None
    summary = pl.read_parquet(PREPROCESSING_SUMMARY)
    totals = summary.select(
        pl.col("nonblank_rows").sum().alias("nonblank_rows"),
        pl.col("eligible_rows").sum().alias("eligible_rows"),
    ).row(0, named=True)
    eligible_rows = int(totals["eligible_rows"])
    nonblank_rows = int(totals["nonblank_rows"])
    return {
        "task_count": summary.height,
        "nonblank_rows": nonblank_rows,
        "eligible_rows": eligible_rows,
        "eligible_rate": eligible_rows / nonblank_rows
        if nonblank_rows
        else None,
    }


def _sampling_summary() -> dict[str, object] | None:
    if not SELECTED_SAMPLE.is_file():
        return None
    selected = pl.read_parquet(SELECTED_SAMPLE)
    payload: dict[str, object] = {
        "selected_generations": selected.height,
        "distinct_tasks": selected.get_column("task_id").n_unique(),
    }
    if SAMPLING_COVERAGE.is_file():
        coverage = pl.read_parquet(SAMPLING_COVERAGE)
        payload["coverage_cells"] = coverage.height
        payload["selected_cells"] = int(coverage.get_column("selected").sum())
    return payload


def _evaluation_summary(paths) -> dict[str, object] | None:  # noqa: ANN001
    if not paths.task_results.is_file():
        return None
    tasks = pl.read_parquet(paths.task_results)
    mean_rate = tasks.get_column("test_success_rate").mean()
    generations = (
        pl.read_parquet(paths.generation_results)
        if paths.generation_results.is_file()
        else None
    )
    extremes = tasks.get_column("observed_extreme").value_counts()
    counts = {
        row["observed_extreme"]: int(row["count"])
        for row in extremes.iter_rows(named=True)
    }
    payload: dict[str, object] = {
        "task_count": tasks.height,
        "observed_extremes": counts,
        "mean_task_success_rate": (
            float(mean_rate) if isinstance(mean_rate, int | float) else None
        ),
    }
    if generations is not None:
        payload["generation_count"] = generations.height
        payload["complete_generations"] = int(
            generations.get_column("evaluation_complete").sum()
        )
        payload["passed_generations"] = int(
            generations.filter(pl.col("generation_passed") == True).height  # noqa: E712
        )
    return payload


def export_baseline(
    name: str,
    *,
    settings=None,  # noqa: ANN001
) -> dict[str, object]:
    if settings is None:
        settings = parse_evaluation_args(None)
    export_dir = baseline_directory(name)
    logs_dir = export_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = run_directory_path()
    paths = evaluation_paths(settings)
    bundle_dir = generation_corpus_bundle_path()
    manifest_summary = load_manifest_summary(bundle_dir)

    stages = _stage_status(run_dir, paths)
    payload: dict[str, object] = {
        "baseline_name": name,
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "git_rev": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "corpus_bundle": str(bundle_dir),
        "manifest_sha256": manifest_sha256(bundle_dir),
        "manifest_summary": manifest_summary,
        "run_directory": str(run_dir),
        "evaluation_settings": {
            "worker_count": settings.worker_count,
            "timeout_seconds": settings.timeout_seconds,
            "evaluation_root": str(paths.root),
        },
        "stages_completed": [stage for stage, done in stages.items() if done],
        "preprocessing": _preprocessing_summary(),
        "sampling": _sampling_summary(),
        "evaluation": _evaluation_summary(paths),
    }

    copied_logs: list[str] = []
    run_log = logs_dir / "run.log"
    if run_log.is_file():
        copied_logs.append("run.log")
    for source, destination_name in (
        (PREPROCESS_LOG, "01_preprocess.log"),
        (SAMPLING_LOG, "02_sample.log"),
        (paths.evaluation_log, "03_evaluate.log"),
        (paths.summary_log, "04_summarize.log"),
    ):
        if _copy_if_exists(source, logs_dir / destination_name):
            copied_logs.append(destination_name)
    payload["copied_logs"] = copied_logs

    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (export_dir / "corpus_identity.json").write_text(
        json.dumps(
            {
                "corpus_bundle": str(bundle_dir),
                "manifest_sha256": payload["manifest_sha256"],
                "manifest_summary": manifest_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (export_dir / "summary.txt").write_text(
        _format_summary_text(payload),
        encoding="utf-8",
    )
    return payload


def _format_summary_text(payload: dict[str, object]) -> str:
    stages_completed = payload.get("stages_completed")
    stage_label = (
        ", ".join(str(stage) for stage in stages_completed)
        if isinstance(stages_completed, list)
        else ""
    )
    lines = [
        f"baseline: {payload['baseline_name']}",
        f"exported_at: {payload['exported_at']}",
        f"git_rev: {payload.get('git_rev')}",
        f"git_branch: {payload.get('git_branch')}",
        f"corpus_bundle: {payload['corpus_bundle']}",
        f"manifest_sha256: {payload['manifest_sha256']}",
        f"run_directory: {payload['run_directory']}",
        f"stages_completed: {stage_label}",
    ]
    preprocessing = payload.get("preprocessing")
    if isinstance(preprocessing, dict):
        lines.append(
            "preprocessing: "
            f"{preprocessing.get('eligible_rows')} eligible / "
            f"{preprocessing.get('nonblank_rows')} nonblank "
            f"({preprocessing.get('task_count')} tasks)"
        )
    sampling = payload.get("sampling")
    if isinstance(sampling, dict):
        lines.append(
            "sampling: "
            f"{sampling.get('selected_generations')} selected generations "
            f"across {sampling.get('distinct_tasks')} tasks"
        )
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        mean_rate = evaluation.get("mean_task_success_rate")
        mean_rate_label = (
            f"{mean_rate:.4f}" if isinstance(mean_rate, int | float) else "n/a"
        )
        lines.append(
            "evaluation: "
            f"{evaluation.get('complete_generations', 'n/a')}/"
            f"{evaluation.get('generation_count', 'n/a')} complete generations, "
            f"mean task success rate {mean_rate_label}"
        )
        extremes = evaluation.get("observed_extremes")
        if isinstance(extremes, dict):
            lines.append(f"observed task extremes: {extremes}")
    copied_logs = payload.get("copied_logs")
    if isinstance(copied_logs, list):
        lines.append(f"logs: {', '.join(str(name) for name in copied_logs)}")
    return "\n".join(lines) + "\n"


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline_name",
        help="baseline directory name under scripts/verification/task_difficulty/baseline/",
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments, remaining = _parse_args(argv)
    settings = parse_evaluation_args(None, remaining)
    payload = export_baseline(arguments.baseline_name, settings=settings)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
