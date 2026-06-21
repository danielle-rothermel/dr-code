"""Generate fresh decoder attempts via dr-providers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from dr_code.datasets.export import write_attempts
from dr_code.datasets.stats import format_summary, summarize_attempts
from dr_code.generation.batch import generate_attempts
from dr_code.generation.profiles import default_profiles_path, list_profile_ids
from dr_code.generation.run_config import GenerationRunConfig

app = typer.Typer(add_completion=False)


def _parse_task_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    ids = [part.strip() for part in value.split(",") if part.strip()]
    return ids or None


@app.command()
def main(
    list_profiles: bool = typer.Option(
        False,
        "--list-profiles",
        help="Print valid OpenRouter profile ids and exit",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="OpenRouter profile id from configs/openrouter_profiles.yaml",
    ),
    profiles_path: Path = typer.Option(
        default_profiles_path(),
        "--profiles-path",
        help="Path to OpenRouter profiles YAML",
    ),
    task_ids: str | None = typer.Option(
        None,
        "--task-ids",
        help="Comma-separated HumanEval task ids",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Cap number of tasks generated",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run id for provenance (default: fresh-{uuid8})",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Export path (.parquet or .jsonl)",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Optional completion token limit",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Print summary stats after export",
    ),
) -> None:
    """Generate fresh_stub AttemptRecord exports via dr-providers."""
    if list_profiles:
        for profile_id in list_profile_ids(profiles_path):
            typer.echo(profile_id)
        return

    if profile is None:
        msg = "Provide --profile or --list-profiles"
        raise typer.BadParameter(msg)

    resolved_run_id = run_id or f"fresh-{uuid4().hex[:8]}"
    resolved_output = output or Path(
        f"exports/attempts/fresh_{resolved_run_id}.parquet"
    )
    config = GenerationRunConfig(
        run_id=resolved_run_id,
        profile_id=profile,
        profiles_path=profiles_path,
        task_ids=_parse_task_ids(task_ids),
        limit=limit,
        max_tokens=max_tokens,
    )
    records = generate_attempts(config)
    write_attempts(records, resolved_output)
    typer.echo(f"Wrote {len(records)} records to {resolved_output}")
    if stats:
        summary = summarize_attempts(records)
        typer.echo(format_summary(summary))


if __name__ == "__main__":
    app()
