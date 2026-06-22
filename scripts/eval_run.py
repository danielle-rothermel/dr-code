"""Evaluation run lifecycle CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dr_code.pipeline.lifecycle import (
    DEFAULT_HANDLERS_MODULE,
    DEFAULT_WORKERS,
    EvalStatusResult,
    WaitEvalRunResult,
    echo_proof_summary,
    echo_run_metadata,
    export_eval_run,
    get_eval_status,
    init_eval_run,
    preflight_eval_run,
    run_eval_once,
    seed_eval_run,
    start_eval_workers,
    stop_eval_workers,
    wait_for_eval_run,
)
from dr_code.pipeline.seed import DEFAULT_DUMP_DIR, DEFAULT_PROOF_INDICES

app = typer.Typer(add_completion=False)

_DEFAULT_TASK_INDICES = ",".join(str(index) for index in DEFAULT_PROOF_INDICES)


def _parse_task_indices(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        msg = "task-indices must contain at least one index"
        raise typer.BadParameter(msg)
    return [int(part) for part in parts]


def _echo_status_summary(result: EvalStatusResult | WaitEvalRunResult) -> None:
    status = result.status
    typer.echo(
        f"run_id={result.run_id} terminals={status.terminal_jobs}/"
        f"{status.expected_jobs} complete={status.is_complete}",
    )
    for stage in status.stages:
        typer.echo(
            f"  {stage.stage}: completed={stage.completed_jobs}/"
            f"{stage.expected_jobs} in_flight={stage.in_flight_jobs} "
            f"ready={stage.input_queue.ready_messages} "
            f"workers={len(stage.workers)}",
        )


@app.command()
def preflight(
    dump_dir: Annotated[
        Path,
        typer.Option("--dump-dir", help="Root of pool dump artifacts"),
    ] = DEFAULT_DUMP_DIR,
    task_indices: Annotated[
        str,
        typer.Option("--task-indices", help="Comma-separated task indices"),
    ] = _DEFAULT_TASK_INDICES,
    require_docker: Annotated[bool, typer.Option("--require-docker")] = False,
    require_dump: Annotated[
        bool, typer.Option("--require-dump/--no-require-dump")
    ] = True,
) -> None:
    """Check local services and seed inputs."""
    report = preflight_eval_run(
        dump_dir=dump_dir,
        task_indices=_parse_task_indices(task_indices),
        require_docker=require_docker,
        require_dump=require_dump,
    ).report
    for check in report.checks:
        typer.echo(f"preflight ok: {check}")
    report.raise_if_failed()


@app.command()
def init(
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    workers: Annotated[str, typer.Option("--workers")] = DEFAULT_WORKERS,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Create an Evaluation run manifest and queues."""
    result = init_eval_run(run_id=run_id, workers=workers, overwrite=overwrite)
    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"workers={result.workers_by_stage}")


@app.command()
def seed(
    run_id: Annotated[str, typer.Option("--run-id")],
    attempts: Annotated[
        Path | None,
        typer.Option("--attempts", help="AttemptRecord parquet export"),
    ] = None,
    dump_dir: Annotated[
        Path | None,
        typer.Option("--dump-dir", help="Root of pool dump artifacts"),
    ] = None,
    task_indices: Annotated[
        str,
        typer.Option("--task-indices", help="Comma-separated task indices"),
    ] = _DEFAULT_TASK_INDICES,
    limit_per_task: Annotated[
        int | None, typer.Option("--limit-per-task")
    ] = None,
) -> None:
    """Seed Decoder attempts into an initialized Evaluation run."""
    result = seed_eval_run(
        run_id=run_id,
        attempts_path=attempts,
        dump_dir=dump_dir,
        task_indices=_parse_task_indices(task_indices) if dump_dir else None,
        limit_per_task=limit_per_task,
    )
    typer.echo(f"run_id={result.run_id} expected_jobs={result.expected_jobs}")


@app.command()
def start(
    run_id: Annotated[str, typer.Option("--run-id")],
    workers: Annotated[str, typer.Option("--workers")] = DEFAULT_WORKERS,
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", help="Stage to start; repeat for more"),
    ] = None,
    handlers_module: Annotated[
        str,
        typer.Option("--handlers-module"),
    ] = DEFAULT_HANDLERS_MODULE,
) -> None:
    """Start detached stage workers."""
    result = start_eval_workers(
        run_id=run_id,
        workers=workers,
        stages=stage,
        handlers_module=handlers_module,
    )
    typer.echo(
        f"run_id={result.run_id} pids={','.join(map(str, result.pids))}"
    )


@app.command()
def stop(
    run_id: Annotated[str, typer.Option("--run-id")],
    stage: Annotated[
        list[str] | None,
        typer.Option("--stage", help="Stage to stop; repeat for more"),
    ] = None,
    worker_id: Annotated[
        list[str] | None,
        typer.Option("--worker-id", help="Worker id to stop; repeat for more"),
    ] = None,
) -> None:
    """Request selected workers to stop."""
    result = stop_eval_workers(
        run_id=run_id,
        stages=stage,
        worker_ids=worker_id,
    )
    typer.echo(f"run_id={result.run_id} stop_requested={len(result.workers)}")


@app.command()
def wait(
    run_id: Annotated[str, typer.Option("--run-id")],
    target: Annotated[str, typer.Option("--target")] = "terminal",
    timeout: Annotated[float | None, typer.Option("--timeout")] = None,
    poll_interval: Annotated[float, typer.Option("--poll-interval")] = 1.0,
) -> None:
    """Wait for a run target; terminal wait records terminal events."""
    result = wait_for_eval_run(
        run_id=run_id,
        target=target,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    _echo_status_summary(result)


@app.command()
def status(
    run_id: Annotated[str, typer.Option("--run-id")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print Evaluation run status."""
    result = get_eval_status(run_id)
    if json_output:
        typer.echo(result.status.model_dump_json())
    else:
        _echo_status_summary(result)


@app.command()
def export(
    run_id: Annotated[str, typer.Option("--run-id")],
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "exports/runs"
    ),
) -> None:
    """Export derived artifacts for an Evaluation run."""
    result = export_eval_run(run_id=run_id, output_root=output_root)
    typer.echo(f"run_id={result.run_id} exports={result.export_paths.run_dir}")


@app.command()
def run(
    mode: Annotated[
        str,
        typer.Option("--mode", help="in-process or detached"),
    ] = "in-process",
    attempts: Annotated[
        Path | None,
        typer.Option("--attempts", help="AttemptRecord parquet export"),
    ] = None,
    dump_dir: Annotated[
        Path,
        typer.Option("--dump-dir", help="Root of pool dump artifacts"),
    ] = DEFAULT_DUMP_DIR,
    task_indices: Annotated[
        str,
        typer.Option("--task-indices", help="Comma-separated task indices"),
    ] = _DEFAULT_TASK_INDICES,
    limit_per_task: Annotated[
        int | None, typer.Option("--limit-per-task")
    ] = None,
    workers: Annotated[str, typer.Option("--workers")] = DEFAULT_WORKERS,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    handlers_module: Annotated[
        str,
        typer.Option("--handlers-module"),
    ] = DEFAULT_HANDLERS_MODULE,
    completion_timeout: Annotated[
        float,
        typer.Option("--completion-timeout"),
    ] = 28800.0,
    output_root: Annotated[Path, typer.Option("--output-root")] = Path(
        "exports/runs"
    ),
    skip_preflight: Annotated[bool, typer.Option("--skip-preflight")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Preflight, seed, execute, wait, export, and report."""
    indices = _parse_task_indices(task_indices)
    result = run_eval_once(
        mode=mode,
        attempts_path=attempts,
        dump_dir=None if attempts else dump_dir,
        task_indices=None if attempts else indices,
        limit_per_task=limit_per_task,
        workers=workers,
        run_id=run_id,
        handlers_module=handlers_module,
        completion_timeout=completion_timeout,
        output_root=output_root,
        skip_preflight=skip_preflight,
        overwrite=overwrite,
    ).pipeline_result
    echo_run_metadata(
        run_id=result.run_id,
        expected_jobs=result.expected_jobs,
        mode=mode,
        workers=workers,
    )
    echo_proof_summary(result)


if __name__ == "__main__":
    app()
