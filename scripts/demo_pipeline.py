"""In-process pipeline demo on a handful of pool samples."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.datasets.pool_loader import load_pool_dedup_jsonl
from dr_code.pipeline.jobs import stamp_run_id
from dr_code.pipeline.preflight import run_preflight
from dr_code.pipeline.runner import (
    DEFAULT_WORKERS,
    echo_proof_summary,
    echo_run_metadata,
    new_run_id,
    run_eval_pipeline,
)

app = typer.Typer(add_completion=False)

_DEFAULT_FIXTURE_DEDUP = Path(
    "tests/fixtures/pool/human_eval-0-decode-dedup.jsonl"
)
_DEFAULT_MONGODB_URL = "mongodb://localhost:27017/dr_queues"


def _section(title: str) -> None:
    typer.echo("")
    typer.echo(f"=== {title} ===")


def _load_demo_attempts(
    *,
    dedup_path: Path,
    limit: int,
    run_id: str,
) -> list:
    records = load_pool_dedup_jsonl(dedup_path, limit=limit)
    return stamp_run_id(records, run_id)


def _print_mongo_hints(run_id: str, sample_id: str | None) -> None:
    _section("MongoDB — inspect pipeline run")
    typer.echo(f"MONGODB_URL={_DEFAULT_MONGODB_URL}")
    typer.echo("")
    typer.echo("# Count terminal events for this run")
    typer.echo(
        f"mongosh {_DEFAULT_MONGODB_URL} \\\n"
        f"  --eval 'db.pipeline_events.countDocuments("
        f'{{run_id: "{run_id}", event: "terminal"}}'
        f")'"
    )
    typer.echo("")
    typer.echo("# Count eval_results for this run")
    typer.echo(
        f"mongosh {_DEFAULT_MONGODB_URL} \\\n"
        f"  --eval 'db.eval_results.countDocuments("
        f'{{run_id: "{run_id}"}}'
        f")'"
    )
    if sample_id is not None:
        typer.echo("")
        typer.echo("# Sample eval result")
        typer.echo(
            f"mongosh {_DEFAULT_MONGODB_URL} \\\n"
            f"  --eval 'db.eval_results.findOne("
            f'{{run_id: "{run_id}", sample_id: "{sample_id}"}}'
            f")'"
        )


@app.command()
def main(
    limit: int = typer.Option(3, "--limit", help="Number of dedup samples"),
    dedup_path: Path | None = typer.Option(
        None,
        "--dedup-path",
        help="Dedup JSONL path (default: tests/fixtures/pool)",
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
    workers: str = typer.Option(DEFAULT_WORKERS, "--workers"),
    completion_timeout: float = typer.Option(600.0, "--completion-timeout"),
    skip_preflight: bool = typer.Option(False, "--skip-preflight"),
) -> None:
    """Run in-process parse→test pipeline on a small pool sample set."""
    resolved_run_id = run_id or new_run_id("demo")
    resolved_dedup = dedup_path or _DEFAULT_FIXTURE_DEDUP
    if not resolved_dedup.is_file():
        msg = f"Dedup file not found: {resolved_dedup}"
        raise typer.BadParameter(msg)

    if not skip_preflight:
        report = run_preflight(require_dump=False)
        for check in report.checks:
            typer.echo(f"preflight ok: {check}")
        report.raise_if_failed()

    attempts = _load_demo_attempts(
        dedup_path=resolved_dedup,
        limit=limit,
        run_id=resolved_run_id,
    )
    echo_run_metadata(
        run_id=resolved_run_id,
        expected_jobs=len(attempts),
        mode="in-process",
        workers=workers,
    )

    result = run_eval_pipeline(
        attempts,
        run_id=resolved_run_id,
        mode="in-process",
        workers=workers,
        completion_timeout=completion_timeout,
    )

    _section("Proof summary")
    echo_proof_summary(result)

    sample_id = attempts[0].sample_id if attempts else None
    _print_mongo_hints(resolved_run_id, sample_id)


if __name__ == "__main__":
    app()
