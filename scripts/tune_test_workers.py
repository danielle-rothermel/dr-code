"""Live test-worker throughput sweep for in-flight eval runs."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.pipeline.runner import DEFAULT_HANDLERS_MODULE
from dr_code.pipeline.tune import (
    count_stage_completions,
    count_terminals,
    format_sweep_table,
    run_sweep,
)
from dr_queues import MongoRunStore, list_workers

app = typer.Typer(add_completion=False)

_DEFAULT_OUTPUT_ROOT = Path("exports/runs")


@app.command()
def main(
    run_id: str = typer.Option(..., "--run-id"),
    start_workers: int = typer.Option(2, "--start-workers"),
    multiplier: int = typer.Option(2, "--multiplier"),
    window_seconds: float = typer.Option(60.0, "--window-seconds"),
    warmup_seconds: float = typer.Option(15.0, "--warmup-seconds"),
    max_workers: int = typer.Option(16, "--max-workers"),
    stop_threshold: float = typer.Option(0.10, "--stop-threshold"),
    min_samples_in_window: int = typer.Option(5, "--min-samples-in-window"),
    handlers_module: str = typer.Option(
        DEFAULT_HANDLERS_MODULE,
        "--handlers-module",
    ),
    output_root: Path = typer.Option(_DEFAULT_OUTPUT_ROOT, "--output-root"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    apply_best: bool = typer.Option(True, "--apply-best/--no-apply-best"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
) -> None:
    """Sweep test worker counts on a live detached eval run."""
    store = MongoRunStore()
    try:
        expected_jobs = store.expected_job_count(run_id)
    finally:
        store.close()

    if not skip_preflight:
        _run_preflight(run_id, expected_jobs=expected_jobs)

    def _on_step(step: object) -> None:
        from dr_code.pipeline.tune import SweepStepResult

        assert isinstance(step, SweepStepResult)
        typer.echo(
            f"workers={step.workers} rate={step.samples_per_second:.3f}/s "
            f"delta={step.terminals_after - step.terminals_before} "
            f"reliable={step.reliable}"
        )

    report = run_sweep(
        run_id=run_id,
        expected_jobs=expected_jobs,
        start_workers=start_workers,
        multiplier=multiplier,
        window_seconds=window_seconds,
        warmup_seconds=warmup_seconds,
        max_workers=max_workers,
        stop_threshold=stop_threshold,
        min_samples_in_window=min_samples_in_window,
        handlers_module=handlers_module,
        dry_run=dry_run,
        apply_best=apply_best,
        on_step=_on_step,
    )

    out_dir = output_root / run_id
    report_path = report.write_json(out_dir / "tune_report.json")
    typer.echo("")
    typer.echo(format_sweep_table(report))
    typer.echo(f"tune_report={report_path}")


def _run_preflight(run_id: str, *, expected_jobs: int) -> None:
    terminals = count_terminals(run_id)
    parse_done = count_stage_completions(run_id, "parse")
    test_workers = list_workers(run_id, stage="test")

    typer.echo(f"preflight terminals={terminals}/{expected_jobs}")
    typer.echo(f"preflight parse_completions={parse_done}/{expected_jobs}")
    typer.echo(
        f"preflight test_worker_pids={[worker.pid for worker in test_workers]}"
    )

    if terminals >= expected_jobs:
        msg = "Run already complete; nothing to tune."
        raise typer.BadParameter(msg)
    if parse_done < expected_jobs:
        typer.echo(
            "warning: parse stage not fully complete; sweep still targets test workers",
        )


if __name__ == "__main__":
    app()
