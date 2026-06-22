"""Evaluation run lifecycle interface."""

from __future__ import annotations

from pathlib import Path

from dr_queues import (
    MongoRunStore,
    RunManifest,
    RunStatus,
    WorkerRecord,
    attach_run_queues,
    get_run_status,
    parse_workers_arg,
    seed_run,
    setup_run_queues,
    spawn_all_stage_workers,
    start_stage_workers,
    stop_workers,
    wait_for_run,
)

from dr_code.datasets.export import read_attempts
from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.pipeline.definition import build_eval_pipeline
from dr_code.pipeline.export import RunExportPaths, export_run_artifacts
from dr_code.pipeline.handlers import registry
from dr_code.pipeline.jobs import build_seed_jobs
from dr_code.pipeline.runner import (
    DEFAULT_HANDLERS_MODULE,
    DEFAULT_WORKERS,
    PipelineRunResult,
    new_run_id,
    run_eval_pipeline,
)
from dr_code.pipeline.seed import load_proof_attempts


class InitEvalRunResult(FrozenModel):
    """Initialized Evaluation run manifest."""

    run_id: str
    manifest: RunManifest
    workers_by_stage: dict[str, int]


class SeedEvalRunResult(FrozenModel):
    """Seeded Evaluation run jobs."""

    run_id: str
    expected_jobs: int


class StartEvalWorkersResult(FrozenModel):
    """Started Evaluation run worker processes."""

    run_id: str
    pids: list[int]


class StopEvalWorkersResult(FrozenModel):
    """Stopped Evaluation run workers."""

    run_id: str
    workers: list[WorkerRecord]


class WaitEvalRunResult(FrozenModel):
    """Wait result for an Evaluation run."""

    run_id: str
    status: RunStatus


class EvalStatusResult(FrozenModel):
    """Current Evaluation run status."""

    run_id: str
    status: RunStatus


class ExportEvalRunResult(FrozenModel):
    """Exported Evaluation run artifacts."""

    run_id: str
    export_paths: RunExportPaths


class RunEvalOnceResult(FrozenModel):
    """Completed one-shot Evaluation run."""

    run_id: str
    pipeline_result: PipelineRunResult


def init_eval_run(
    *,
    run_id: str | None = None,
    workers: str = DEFAULT_WORKERS,
    run_store: MongoRunStore | None = None,
    overwrite: bool = False,
) -> InitEvalRunResult:
    """Create the dr-queues manifest and queues for an Evaluation run."""
    resolved_run_id = run_id or new_run_id()
    pipeline = build_eval_pipeline(registry)
    workers_by_stage = parse_workers_arg(
        workers,
        pipeline.step_names(),
        default=2,
    )
    manifest = setup_run_queues(
        pipeline=pipeline,
        run_id=resolved_run_id,
        workers_by_stage=workers_by_stage,
        run_store=run_store,
        overwrite=overwrite,
    )
    return InitEvalRunResult(
        run_id=resolved_run_id,
        manifest=manifest,
        workers_by_stage=workers_by_stage,
    )


def seed_eval_run(
    attempts: list[AttemptRecord] | None = None,
    *,
    run_id: str,
    attempts_path: Path | str | None = None,
    dump_dir: Path | str | None = None,
    task_indices: list[int] | tuple[int, ...] | None = None,
    limit_per_task: int | None = None,
    run_store: MongoRunStore | None = None,
) -> SeedEvalRunResult:
    """Publish Evaluation run attempts to the parse stage."""
    manifest = attach_run_queues(
        run_id=run_id,
        pipeline=build_eval_pipeline(registry),
        run_store=run_store,
    )
    records = _load_seed_attempts(
        attempts=attempts,
        attempts_path=attempts_path,
        dump_dir=dump_dir,
        task_indices=task_indices,
        run_id=run_id,
        limit_per_task=limit_per_task,
    )
    jobs = build_seed_jobs(records, run_id=run_id)
    seed_run(manifest, jobs, run_store=run_store)
    return SeedEvalRunResult(run_id=run_id, expected_jobs=len(jobs))


def _load_seed_attempts(
    *,
    attempts: list[AttemptRecord] | None,
    attempts_path: Path | str | None,
    dump_dir: Path | str | None,
    task_indices: list[int] | tuple[int, ...] | None,
    run_id: str,
    limit_per_task: int | None,
) -> list[AttemptRecord]:
    sources = sum(
        source is not None for source in (attempts, attempts_path, dump_dir)
    )
    if sources != 1:
        msg = "Provide exactly one seed source: attempts, attempts_path, or dump_dir."
        raise ValueError(msg)
    if attempts is not None:
        records = attempts
    elif attempts_path is not None:
        records = read_attempts(attempts_path)
    else:
        if dump_dir is None:
            msg = "dump_dir is required when seeding from pool replay inputs."
            raise ValueError(msg)
        if task_indices is None:
            msg = "task_indices is required when seeding from dump_dir."
            raise ValueError(msg)
        records = load_proof_attempts(
            dump_dir,
            task_indices,
            run_id=run_id,
            limit_per_task=limit_per_task,
        )
    return [record.model_copy(update={"run_id": run_id}) for record in records]


def start_eval_workers(
    *,
    run_id: str,
    workers: str = DEFAULT_WORKERS,
    handlers_module: str = DEFAULT_HANDLERS_MODULE,
    stages: list[str] | tuple[str, ...] | None = None,
    run_store: MongoRunStore | None = None,
) -> StartEvalWorkersResult:
    """Start dr-queues workers for an Evaluation run."""
    pipeline = build_eval_pipeline(registry)
    manifest = attach_run_queues(
        run_id=run_id,
        pipeline=pipeline,
        run_store=run_store,
    )
    workers_by_stage = parse_workers_arg(
        workers,
        pipeline.step_names(),
        default=2,
    )
    if stages is None:
        processes = spawn_all_stage_workers(
            manifest=manifest,
            workers_by_stage=workers_by_stage,
            handlers_module=handlers_module,
        )
    else:
        processes = [
            start_stage_workers(
                run_id=run_id,
                stage=stage,
                workers=workers_by_stage[stage],
                handlers_module=handlers_module,
                run_store=run_store,
            )
            for stage in stages
        ]
    return StartEvalWorkersResult(
        run_id=run_id,
        pids=[process.pid for process in processes],
    )


def stop_eval_workers(
    *,
    run_id: str,
    worker_id: str | None = None,
    stage: str | None = None,
    run_store: MongoRunStore | None = None,
) -> StopEvalWorkersResult:
    """Request Evaluation run workers to stop."""
    workers = stop_workers(
        run_id=run_id,
        worker_id=worker_id,
        stage=stage,
        run_store=run_store,
    )
    return StopEvalWorkersResult(run_id=run_id, workers=workers)


def wait_for_eval_run(
    *,
    run_id: str,
    target: str = "terminal",
    timeout: float | None = None,
    poll_interval: float = 1.0,
    run_store: MongoRunStore | None = None,
) -> WaitEvalRunResult:
    """Wait for an Evaluation run to reach a dr-queues target."""
    status = wait_for_run(
        run_id,
        target=target,
        timeout=timeout,
        poll_interval=poll_interval,
        run_store=run_store,
    )
    return WaitEvalRunResult(run_id=run_id, status=status)


def get_eval_status(
    run_id: str,
    *,
    run_store: MongoRunStore | None = None,
) -> EvalStatusResult:
    """Read the current Evaluation run status."""
    status = get_run_status(run_id, run_store=run_store)
    return EvalStatusResult(run_id=run_id, status=status)


def export_eval_run(
    *,
    run_id: str,
    mongo_sink: MongoRunStore | None = None,
    output_root: Path | str = Path("exports/runs"),
) -> ExportEvalRunResult:
    """Export Evaluation run artifacts from persisted run state."""
    export_paths = export_run_artifacts(
        run_id=run_id,
        mongo_sink=mongo_sink,
        output_root=output_root,
    )
    return ExportEvalRunResult(run_id=run_id, export_paths=export_paths)


def run_eval_once(
    attempts: list[AttemptRecord],
    *,
    run_id: str | None = None,
    mode: str = "in-process",
    workers: str = DEFAULT_WORKERS,
    handlers_module: str = DEFAULT_HANDLERS_MODULE,
    completion_timeout: float = 7200.0,
    output_root: Path | str = Path("exports/runs"),
) -> RunEvalOnceResult:
    """Run the existing one-shot Evaluation pipeline."""
    result = run_eval_pipeline(
        attempts,
        run_id=run_id,
        mode=mode,
        workers=workers,
        handlers_module=handlers_module,
        completion_timeout=completion_timeout,
        output_root=output_root,
    )
    return RunEvalOnceResult(run_id=result.run_id, pipeline_result=result)


__all__ = [
    "EvalStatusResult",
    "ExportEvalRunResult",
    "InitEvalRunResult",
    "RunEvalOnceResult",
    "SeedEvalRunResult",
    "StartEvalWorkersResult",
    "StopEvalWorkersResult",
    "WaitEvalRunResult",
    "export_eval_run",
    "get_eval_status",
    "init_eval_run",
    "run_eval_once",
    "seed_eval_run",
    "start_eval_workers",
    "stop_eval_workers",
    "wait_for_eval_run",
]
