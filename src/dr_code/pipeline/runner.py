"""Shared pipeline run orchestration."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from uuid import uuid4

from dr_queues import (
    EventKind,
    MongoRunStore,
    TerminalTap,
    filter_run_events,
    parse_workers_arg,
    run_in_process,
    seed_run,
    setup_run_queues,
    spawn_all_stage_workers,
)

from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.pipeline.definition import build_eval_pipeline
from dr_code.pipeline.export import RunExportPaths, export_run_artifacts
from dr_code.pipeline.handlers import registry
from dr_code.pipeline.jobs import build_seed_jobs
from dr_code.pipeline.report import (
    ProofReport,
    build_proof_report,
    format_proof_summary,
)

DEFAULT_HANDLERS_MODULE = "dr_code.pipeline.handlers"
DEFAULT_WORKERS = "parse=2,test=1"


class PipelineRunResult(FrozenModel):
    """Artifacts from a completed pipeline run."""

    run_id: str
    expected_jobs: int
    terminal_count: int
    wall_seconds: float
    export_paths: RunExportPaths
    proof_report: ProofReport


def new_run_id(prefix: str = "eval") -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def run_eval_pipeline(
    attempts: list[AttemptRecord],
    *,
    run_id: str | None = None,
    mode: str = "in-process",
    workers: str = DEFAULT_WORKERS,
    handlers_module: str = DEFAULT_HANDLERS_MODULE,
    completion_timeout: float = 7200.0,
    output_root: Path | str = Path("exports/runs"),
) -> PipelineRunResult:
    """Seed, execute, export, and report on an eval pipeline run."""
    resolved_run_id = run_id or new_run_id()
    pipeline = build_eval_pipeline(registry)
    workers_by_stage = parse_workers_arg(
        workers, pipeline.step_names(), default=2
    )
    jobs = build_seed_jobs(attempts, run_id=resolved_run_id)
    expected_jobs = len(jobs)

    event_sink = MongoRunStore()
    started = time.perf_counter()
    worker_processes: list[subprocess.Popen[bytes]] = []

    manifest = setup_run_queues(
        pipeline=pipeline,
        run_id=resolved_run_id,
        workers_by_stage=workers_by_stage,
        run_store=event_sink,
    )
    seed_run(manifest, jobs, run_store=event_sink)

    if mode == "in-process":
        run_in_process(
            manifest=manifest,
            pipeline=pipeline,
            workers_by_stage=workers_by_stage,
            run_store=event_sink,
            completion_timeout=completion_timeout,
        )
        terminal_count = expected_jobs
    elif mode == "detached":
        final_stage = manifest.stages[-1]
        tap = TerminalTap(
            completed_queue=final_stage.output_queue,
            run_id=resolved_run_id,
            run_store=event_sink,
        )
        tap.start()
        worker_processes = spawn_all_stage_workers(
            manifest=manifest,
            workers_by_stage=workers_by_stage,
            handlers_module=handlers_module,
        )
        if not tap.wait_for_completion(timeout=completion_timeout):
            _stop_processes(worker_processes)
            event_sink.close()
            msg = "Timed out waiting for detached pipeline completion."
            raise TimeoutError(msg)
        tap.stop()
        tap.join(timeout=5)
        terminal_count = expected_jobs
        _stop_processes(worker_processes)
    else:
        event_sink.close()
        msg = f"Unknown mode {mode!r}; expected in-process or detached"
        raise ValueError(msg)

    wall_seconds = time.perf_counter() - started
    events = filter_run_events(
        event_sink.read_by_run_id(resolved_run_id), resolved_run_id
    )
    export_paths = export_run_artifacts(
        run_id=resolved_run_id,
        attempts=attempts,
        mongo_sink=event_sink,
        output_root=output_root,
    )
    parse_outcomes, test_outcomes = _load_outcomes_from_export(export_paths)
    proof_report = build_proof_report(
        run_id=resolved_run_id,
        attempts=attempts,
        events=events,
        parse_outcomes=parse_outcomes,
        test_outcomes=test_outcomes,
        expected_jobs=expected_jobs,
        terminal_count=terminal_count,
        wall_seconds=wall_seconds,
    )
    report_path = export_paths.run_dir / "proof_report.json"
    proof_report.write_json(report_path)
    event_sink.close()

    terminals = [
        event for event in events if event.event == EventKind.TERMINAL
    ]
    if len(terminals) != expected_jobs:
        msg = (
            f"Terminal count mismatch: {len(terminals)} != {expected_jobs} "
            f"(run_id={resolved_run_id})"
        )
        raise RuntimeError(msg)

    return PipelineRunResult(
        run_id=resolved_run_id,
        expected_jobs=expected_jobs,
        terminal_count=terminal_count,
        wall_seconds=wall_seconds,
        export_paths=export_paths,
        proof_report=proof_report,
    )


def _stop_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def _load_outcomes_from_export(
    export_paths: RunExportPaths,
) -> tuple[list, list]:
    from dr_code.models.outcomes import ParseOutcome, TestOutcome

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
