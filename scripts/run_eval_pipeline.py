"""Beefy eval pipeline driver: seed pool dumps, run parallel workers, report."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.pipeline.preflight import run_preflight
from dr_code.pipeline.runner import (
    DEFAULT_HANDLERS_MODULE,
    echo_proof_summary,
    echo_run_metadata,
    new_run_id,
    run_eval_pipeline,
)
from dr_code.pipeline.seed import DEFAULT_DUMP_DIR, DEFAULT_PROOF_INDICES, load_proof_attempts

app = typer.Typer(add_completion=False)

_DEFAULT_WORKERS = "parse=8,test=2"


def _parse_task_indices(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        msg = "task-indices must contain at least one index"
        raise typer.BadParameter(msg)
    return [int(part) for part in parts]


@app.command()
def main(
    mode: str = typer.Option(
        "in-process",
        "--mode",
        help="in-process or detached",
    ),
    dump_dir: Path = typer.Option(
        DEFAULT_DUMP_DIR,
        "--dump-dir",
        help="Root of dr-llm pool dump artifacts",
    ),
    task_indices: str = typer.Option(
        ",".join(str(index) for index in DEFAULT_PROOF_INDICES),
        "--task-indices",
        help="Comma-separated HumanEval task indices",
    ),
    limit_per_task: int | None = typer.Option(
        None,
        "--limit-per-task",
        help="Cap dedup rows per task (debug)",
    ),
    workers: str = typer.Option(_DEFAULT_WORKERS, "--workers"),
    run_id: str | None = typer.Option(None, "--run-id"),
    handlers_module: str = typer.Option(
        DEFAULT_HANDLERS_MODULE,
        "--handlers-module",
    ),
    completion_timeout: float = typer.Option(
        28800.0,
        "--completion-timeout",
        help="Seconds to wait for pipeline completion",
    ),
    output_root: Path = typer.Option(
        Path("exports/runs"),
        "--output-root",
    ),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
) -> None:
    """Seed and run the eval pipeline on pool dump artifacts."""
    if mode not in {"in-process", "detached"}:
        msg = f"Unknown mode {mode!r}; expected in-process or detached"
        raise typer.BadParameter(msg)

    indices = _parse_task_indices(task_indices)
    resolved_run_id = run_id or new_run_id("proof")

    if not skip_preflight:
        report = run_preflight(dump_dir=dump_dir, task_indices=indices)
        for check in report.checks:
            typer.echo(f"preflight ok: {check}")
        report.raise_if_failed()

    attempts = load_proof_attempts(
        dump_dir,
        indices,
        run_id=resolved_run_id,
        limit_per_task=limit_per_task,
    )
    echo_run_metadata(
        run_id=resolved_run_id,
        expected_jobs=len(attempts),
        mode=mode,
        workers=workers,
    )

    result = run_eval_pipeline(
        attempts,
        run_id=resolved_run_id,
        mode=mode,
        workers=workers,
        handlers_module=handlers_module,
        completion_timeout=completion_timeout,
        output_root=output_root,
    )
    echo_proof_summary(result)


if __name__ == "__main__":
    app()
