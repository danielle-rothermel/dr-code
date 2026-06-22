"""Evaluation run lifecycle interface."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from dr_queues import (
    EventKind,
    MongoRunStore,
    RunManifest,
    RunStatus,
    WorkerRecord,
    attach_run_queues,
    filter_run_events,
    get_run_status,
    parse_workers_arg,
    run_in_process,
    seed_run,
    setup_run_queues,
    start_stage_workers,
    stop_workers,
    wait_for_run,
)

from dr_code.datasets.export import read_attempts
from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.pipeline.constants import DEFAULT_HANDLERS_MODULE, DEFAULT_WORKERS
from dr_code.pipeline.definition import build_eval_pipeline
from dr_code.pipeline.export import (
    RunExportPaths,
    export_run_artifacts,
    wall_seconds_from_events,
)
from dr_code.pipeline.handlers import registry
from dr_code.pipeline.jobs import build_seed_jobs
from dr_code.pipeline.preflight import PreflightReport, run_preflight
from dr_code.pipeline.report import (
    ProofReport,
    build_proof_report,
    format_proof_summary,
)
from dr_code.pipeline.seed import load_proof_attempts

IN_PROCESS_MODE = "in-process"
DETACHED_MODE = "detached"


class PreflightEvalRunResult(FrozenModel):
    """Evaluation run pre-flight checks."""

    report: PreflightReport


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


class PipelineRunResult(FrozenModel):
    """Artifacts from a completed pipeline run."""

    run_id: str
    expected_jobs: int
    terminal_count: int
    wall_seconds: float
    export_paths: RunExportPaths
    proof_report: ProofReport


class RunEvalOnceResult(FrozenModel):
    """Completed one-shot Evaluation run."""

    run_id: str
    pipeline_result: PipelineRunResult


def new_run_id(prefix: str = "eval") -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def preflight_eval_run(
    *,
    dump_dir: Path | str,
    task_indices: list[int] | tuple[int, ...],
    require_docker: bool = False,
    require_dump: bool = True,
) -> PreflightEvalRunResult:
    """Run Evaluation run pre-flight checks."""
    report = run_preflight(
        dump_dir=dump_dir,
        task_indices=task_indices,
        require_docker=require_docker,
        require_dump=require_dump,
    )
    return PreflightEvalRunResult(report=report)


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
    selected_stages = (
        list(stages)
        if stages is not None
        else [stage.name for stage in reversed(manifest.stages)]
    )
    processes = [
        start_stage_workers(
            run_id=run_id,
            stage=stage,
            workers=workers_by_stage[stage],
            handlers_module=handlers_module,
            run_store=run_store,
        )
        for stage in selected_stages
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
    worker_ids: list[str] | tuple[str, ...] | None = None,
    stages: list[str] | tuple[str, ...] | None = None,
    run_store: MongoRunStore | None = None,
) -> StopEvalWorkersResult:
    """Request Evaluation run workers to stop."""
    selected_worker_ids = list(
        worker_ids or ([worker_id] if worker_id else [])
    )
    selected_stages = list(stages or ([stage] if stage else []))
    if selected_worker_ids and selected_stages:
        msg = "Stop by worker_id or stage, not both."
        raise ValueError(msg)
    if selected_worker_ids:
        workers = [
            worker
            for selected_worker_id in selected_worker_ids
            for worker in stop_workers(
                run_id=run_id,
                worker_id=selected_worker_id,
                run_store=run_store,
            )
        ]
    elif selected_stages:
        workers = [
            worker
            for selected_stage in selected_stages
            for worker in stop_workers(
                run_id=run_id,
                stage=selected_stage,
                run_store=run_store,
            )
        ]
    else:
        workers = stop_workers(run_id=run_id, run_store=run_store)
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
    attempts: list[AttemptRecord] | None = None,
    *,
    run_id: str | None = None,
    mode: str = IN_PROCESS_MODE,
    workers: str = DEFAULT_WORKERS,
    handlers_module: str = DEFAULT_HANDLERS_MODULE,
    completion_timeout: float = 7200.0,
    output_root: Path | str = Path("exports/runs"),
    attempts_path: Path | str | None = None,
    dump_dir: Path | str | None = None,
    task_indices: list[int] | tuple[int, ...] | None = None,
    limit_per_task: int | None = None,
    skip_preflight: bool = False,
    overwrite: bool = False,
) -> RunEvalOnceResult:
    """Preflight, seed, execute, wait, export, and report on an eval run."""
    if mode not in {IN_PROCESS_MODE, DETACHED_MODE}:
        msg = f"Unknown mode {mode!r}; expected in-process or detached"
        raise ValueError(msg)

    resolved_run_id = run_id or new_run_id("proof")
    if not skip_preflight:
        preflight_eval_run(
            dump_dir=dump_dir or Path("."),
            task_indices=task_indices or [],
            require_dump=dump_dir is not None,
        ).report.raise_if_failed()
    records = _load_seed_attempts(
        attempts=attempts,
        attempts_path=attempts_path,
        dump_dir=dump_dir,
        task_indices=task_indices,
        run_id=resolved_run_id,
        limit_per_task=limit_per_task,
    )

    store = MongoRunStore()
    try:
        init_result = init_eval_run(
            run_id=resolved_run_id,
            workers=workers,
            run_store=store,
            overwrite=overwrite,
        )
        seed_result = seed_eval_run(
            records,
            run_id=resolved_run_id,
            run_store=store,
        )
        pipeline = build_eval_pipeline(registry)
        if mode == IN_PROCESS_MODE:
            run_in_process(
                manifest=init_result.manifest,
                pipeline=pipeline,
                workers_by_stage=init_result.workers_by_stage,
                run_store=store,
                completion_timeout=completion_timeout,
            )
            status = get_run_status(resolved_run_id, run_store=store)
        else:
            try:
                start_eval_workers(
                    run_id=resolved_run_id,
                    workers=workers,
                    handlers_module=handlers_module,
                    stages=["test", "parse"],
                    run_store=store,
                )
                status = wait_for_eval_run(
                    run_id=resolved_run_id,
                    timeout=completion_timeout,
                    run_store=store,
                ).status
                if not status.is_complete:
                    msg = "Timed out waiting for detached pipeline completion."
                    raise TimeoutError(msg)
            finally:
                stop_eval_workers(run_id=resolved_run_id, run_store=store)

        export_paths = export_eval_run(
            run_id=resolved_run_id,
            mongo_sink=store,
            output_root=output_root,
        ).export_paths
        events = filter_run_events(
            store.read_by_run_id(resolved_run_id), resolved_run_id
        )
        wall_seconds = wall_seconds_from_events(events)
        parse_outcomes, test_outcomes = _load_outcomes_from_export(
            export_paths
        )
        terminal_count = len(
            [event for event in events if event.event == EventKind.TERMINAL]
        )
        proof_report = build_proof_report(
            run_id=resolved_run_id,
            attempts=records,
            events=events,
            parse_outcomes=parse_outcomes,
            test_outcomes=test_outcomes,
            expected_jobs=seed_result.expected_jobs,
            terminal_count=terminal_count,
            wall_seconds=wall_seconds,
        )
        if terminal_count != seed_result.expected_jobs:
            msg = (
                f"Terminal count mismatch: {terminal_count} != "
                f"{seed_result.expected_jobs} (run_id={resolved_run_id})"
            )
            raise RuntimeError(msg)
        return RunEvalOnceResult(
            run_id=resolved_run_id,
            pipeline_result=PipelineRunResult(
                run_id=resolved_run_id,
                expected_jobs=seed_result.expected_jobs,
                terminal_count=terminal_count,
                wall_seconds=wall_seconds,
                export_paths=export_paths,
                proof_report=proof_report,
            ),
        )
    finally:
        store.close()


def _load_outcomes_from_export(
    export_paths: RunExportPaths,
) -> tuple[list[ParseOutcome], list[TestOutcome]]:
    parse_outcomes: list[ParseOutcome] = []
    test_outcomes: list[TestOutcome] = []
    if export_paths.parse_jsonl.is_file():
        for line in export_paths.parse_jsonl.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                parse_outcomes.append(ParseOutcome.model_validate_json(line))
    if export_paths.test_jsonl.is_file():
        for line in export_paths.test_jsonl.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                test_outcomes.append(TestOutcome.model_validate_json(line))
    return parse_outcomes, test_outcomes


def echo_run_metadata(
    *,
    run_id: str,
    expected_jobs: int,
    mode: str,
    workers: str,
) -> None:
    import typer

    typer.echo(f"run_id={run_id}")
    typer.echo(f"manifest=mongodb://run_manifests/{run_id}")
    typer.echo(f"expected_jobs={expected_jobs} mode={mode} workers={workers}")


def echo_proof_summary(result: PipelineRunResult) -> None:
    import typer

    typer.echo(format_proof_summary(result.proof_report))
    typer.echo(f"exports={result.export_paths.run_dir}")


__all__ = [
    "DEFAULT_HANDLERS_MODULE",
    "DEFAULT_WORKERS",
    "DETACHED_MODE",
    "EvalStatusResult",
    "ExportEvalRunResult",
    "IN_PROCESS_MODE",
    "InitEvalRunResult",
    "PipelineRunResult",
    "PreflightEvalRunResult",
    "RunEvalOnceResult",
    "SeedEvalRunResult",
    "StartEvalWorkersResult",
    "StopEvalWorkersResult",
    "WaitEvalRunResult",
    "echo_proof_summary",
    "echo_run_metadata",
    "export_eval_run",
    "get_eval_status",
    "init_eval_run",
    "new_run_id",
    "preflight_eval_run",
    "run_eval_once",
    "seed_eval_run",
    "start_eval_workers",
    "stop_eval_workers",
    "wait_for_eval_run",
]
