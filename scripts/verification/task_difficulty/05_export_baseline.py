#!/usr/bin/env python3

"""Export git-tracked baseline run config and results for task-difficulty runs."""

from __future__ import annotations

import argparse
import json
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


def _stage_status(run_dir: Path, paths) -> dict[str, bool]:  # noqa: ANN001
    bundle_path = None
    if paths.run_manifest.is_file():
        try:
            manifest = json.loads(
                paths.run_manifest.read_text(encoding="utf-8")
            )
            if isinstance(manifest, dict):
                stored = manifest.get("bundle_path")
                if isinstance(stored, str) and stored:
                    bundle_path = Path(stored)
        except json.JSONDecodeError:
            bundle_path = None
    return {
        "preprocess": (run_dir / PREPROCESS_LOG.name).is_file()
        and ELIGIBLE_CORPUS.is_file(),
        "sample": (run_dir / SAMPLING_LOG.name).is_file()
        and SELECTED_SAMPLE.is_file(),
        "evaluate": paths.evaluation_log.is_file()
        and paths.run_manifest.is_file()
        and bundle_path is not None
        and (bundle_path / "manifest.json").is_file(),
        "summarize": paths.summary_log.is_file()
        and paths.task_results.is_file(),
    }


def _preprocessing_results() -> dict[str, object] | None:
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
        "task_count": summary.get_column("task_id").n_unique(),
        "nonblank_rows": nonblank_rows,
        "eligible_rows": eligible_rows,
        "eligible_rate": eligible_rows / nonblank_rows
        if nonblank_rows
        else None,
    }


def _sampling_results() -> dict[str, object] | None:
    if not SELECTED_SAMPLE.is_file():
        return None
    selected = pl.read_parquet(SELECTED_SAMPLE)
    payload: dict[str, object] = {
        "selected_generations": selected.height,
        "distinct_tasks": selected.get_column("task_id").n_unique(),
    }
    if "tasks_per_group" in selected.columns:
        tasks_per_group = selected.item(0, "tasks_per_group")
        payload["tasks_per_group"] = (
            "all" if tasks_per_group is None else int(tasks_per_group)
        )
    if SAMPLING_COVERAGE.is_file():
        coverage = pl.read_parquet(SAMPLING_COVERAGE)
        payload["populated_groups"] = coverage.height
        payload["eligible_tasks"] = int(
            coverage.get_column("eligible_tasks").sum()
        )
    return payload


def _evaluation_results(paths) -> dict[str, object] | None:  # noqa: ANN001
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
    export_dir.mkdir(parents=True, exist_ok=True)

    run_dir = run_directory_path()
    paths = evaluation_paths(settings)
    bundle_dir = generation_corpus_bundle_path()
    manifest_summary = load_manifest_summary(bundle_dir)
    stages = _stage_status(run_dir, paths)

    run_config: dict[str, object] = {
        "baseline_name": name,
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
    }
    if paths.run_manifest.is_file():
        try:
            run_manifest = json.loads(
                paths.run_manifest.read_text(encoding="utf-8")
            )
            if isinstance(run_manifest, dict):
                for key in (
                    "attempt_id",
                    "bundle_path",
                    "settings_fingerprint",
                    "status",
                ):
                    if key in run_manifest:
                        run_config[key] = run_manifest[key]
        except json.JSONDecodeError:
            pass
    results: dict[str, object] = {
        "exported_at": datetime.now(tz=UTC).isoformat(),
        "stages_completed": [stage for stage, done in stages.items() if done],
        "preprocessing": _preprocessing_results(),
        "sampling": _sampling_results(),
        "evaluation": _evaluation_results(paths),
    }
    payload = {"run_config": run_config, "results": results}

    (export_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (export_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (export_dir / "results.txt").write_text(
        _format_results_text(run_config, results),
        encoding="utf-8",
    )
    return payload


def _format_results_text(
    run_config: dict[str, object],
    results: dict[str, object],
) -> str:
    stages_completed = results.get("stages_completed")
    stage_label = (
        ", ".join(str(stage) for stage in stages_completed)
        if isinstance(stages_completed, list)
        else ""
    )
    lines = [
        f"baseline: {run_config['baseline_name']}",
        f"exported_at: {results['exported_at']}",
        f"git_rev: {run_config.get('git_rev')}",
        f"git_branch: {run_config.get('git_branch')}",
        f"corpus_bundle: {run_config['corpus_bundle']}",
        f"manifest_sha256: {run_config['manifest_sha256']}",
        f"run_directory: {run_config['run_directory']}",
        f"stages_completed: {stage_label}",
    ]
    preprocessing = results.get("preprocessing")
    if isinstance(preprocessing, dict):
        lines.append(
            "preprocessing: "
            f"{preprocessing.get('eligible_rows')} eligible / "
            f"{preprocessing.get('nonblank_rows')} nonblank "
            f"({preprocessing.get('task_count')} tasks)"
        )
    sampling = results.get("sampling")
    if isinstance(sampling, dict):
        lines.append(
            "sampling: "
            f"{sampling.get('selected_generations')} selected generations "
            f"across {sampling.get('distinct_tasks')} tasks and "
            f"{sampling.get('populated_groups')} groups "
            f"(tasks_per_group={sampling.get('tasks_per_group')})"
        )
    evaluation = results.get("evaluation")
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
