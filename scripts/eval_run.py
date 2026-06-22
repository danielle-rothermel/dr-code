"""Evaluation run lifecycle CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from dr_code.pipeline.lifecycle import (
    DEFAULT_HANDLERS_MODULE,
    DEFAULT_WORKERS,
    EvalStatusResult,
    ListEvalWorkersResult,
    WaitEvalRunResult,
    echo_proof_summary,
    echo_run_metadata,
    export_eval_run,
    get_eval_status,
    init_eval_run,
    list_eval_workers,
    preflight_eval_run,
    replace_eval_workers,
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
    _echo_counts("job_states", getattr(status, "job_state_counts", {}))
    for stage in status.stages:
        workers = list(getattr(stage, "workers", []))
        active_workers = [
            worker
            for worker in workers
            if _worker_status(worker)
            in {
                "running",
                "stop_requested",
            }
        ]
        stale_workers = [
            worker for worker in workers if _worker_status(worker) == "stale"
        ]
        stop_requested_workers = [
            worker
            for worker in workers
            if _worker_status(worker) == "stop_requested"
        ]
        active_concurrency = sum(
            int(getattr(worker, "concurrency", 0)) for worker in active_workers
        )
        input_queue = stage.input_queue
        output_queue = getattr(stage, "output_queue", None)
        output_ready = (
            getattr(output_queue, "ready_messages", 0)
            if output_queue is not None
            else 0
        )
        output_consumers = (
            getattr(output_queue, "consumers", 0)
            if output_queue is not None
            else 0
        )
        typer.echo(
            f"  {stage.stage}: completed={stage.completed_jobs}/"
            f"{stage.expected_jobs} in_flight={stage.in_flight_jobs} "
            f"input_ready={input_queue.ready_messages} "
            f"input_consumers={getattr(input_queue, 'consumers', 0)} "
            f"output_ready={output_ready} "
            f"output_consumers={output_consumers} "
            f"active_workers={len(active_workers)}/{len(workers)} "
            f"active_concurrency={active_concurrency} "
            f"stale_workers={len(stale_workers)} "
            f"stop_requested={len(stop_requested_workers)}",
        )
        _echo_counts(
            f"  {stage.stage} job_states",
            getattr(stage, "job_state_counts", {}),
        )


def _echo_counts(label: str, counts: object) -> None:
    if not isinstance(counts, dict):
        return
    parts = [
        f"{_count_key(key)}={count}" for key, count in counts.items() if count
    ]
    if parts:
        typer.echo(f"{label} {' '.join(parts)}")


def _count_key(key: object) -> str:
    return str(getattr(key, "value", key))


def _worker_status(worker: object) -> str:
    status = getattr(worker, "status", "")
    return str(getattr(status, "value", status))


def _echo_worker_records(result: ListEvalWorkersResult) -> None:
    if not result.workers:
        typer.echo(f"run_id={result.run_id} workers=0")
        return
    for worker in result.workers:
        typer.echo(
            f"worker_id={worker.worker_id} stage={worker.stage} "
            f"status={worker.status} pid={worker.pid} host={worker.host} "
            f"runtime={worker.runtime} concurrency={worker.concurrency}",
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
def replace(
    run_id: Annotated[str, typer.Option("--run-id")],
    stage: Annotated[str, typer.Option("--stage")],
    workers: Annotated[str, typer.Option("--workers")] = DEFAULT_WORKERS,
    handlers_module: Annotated[
        str,
        typer.Option("--handlers-module"),
    ] = DEFAULT_HANDLERS_MODULE,
) -> None:
    """Replace detached workers for one stage."""
    result = replace_eval_workers(
        run_id=run_id,
        stage=stage,
        workers=workers,
        handlers_module=handlers_module,
    )
    typer.echo(f"run_id={result.run_id} stage={result.stage} pid={result.pid}")


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
def workers(
    run_id: Annotated[str, typer.Option("--run-id")],
    stage: Annotated[str | None, typer.Option("--stage")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List Evaluation run worker records."""
    result = list_eval_workers(run_id, stage=stage)
    if json_output:
        typer.echo(
            f"[{','.join(worker.model_dump_json() for worker in result.workers)}]"
        )
        return
    _echo_worker_records(result)


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
