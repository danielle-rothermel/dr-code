"""Import dr-llm pool artifacts into AttemptRecord exports."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.datasets.export import write_attempts
from dr_code.datasets.pool_loader import (
    infer_task_id_from_dedup_path,
    load_pool_dedup_jsonl,
    load_pool_dedup_with_parquet,
    load_pool_parquet,
)
from dr_code.datasets.stats import format_summary, summarize_attempts

app = typer.Typer(add_completion=False)

_DEFAULT_OUTPUT = Path("exports/attempts/pool.parquet")


@app.command()
def main(
    input_path: Path = typer.Option(
        ..., "--input", help="Parquet or dedup JSONL"
    ),
    output: Path = typer.Option(
        _DEFAULT_OUTPUT,
        "--output",
        help="Export path (.parquet or .jsonl)",
    ),
    task_id: str | None = typer.Option(
        None,
        "--task-id",
        help="Task id for dedup JSONL when not inferrable from filename",
    ),
    parquet_join: Path | None = typer.Option(
        None,
        "--parquet-join",
        help="Optional Parquet path for dedup provenance join",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Cap number of rows imported",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        help="Print summary stats after export",
    ),
) -> None:
    """Import pool Parquet or dedup JSONL into unified AttemptRecord export."""
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        records = load_pool_parquet(input_path, limit=limit)
    elif suffix == ".jsonl":
        resolved_task_id = task_id or infer_task_id_from_dedup_path(input_path)
        if parquet_join is not None:
            records = load_pool_dedup_with_parquet(
                input_path,
                parquet_join,
                task_id=resolved_task_id,
                limit=limit,
            )
        else:
            records = load_pool_dedup_jsonl(
                input_path,
                task_id=resolved_task_id,
                limit=limit,
            )
    else:
        msg = f"Unsupported input format: {suffix}"
        raise typer.BadParameter(msg)

    write_attempts(records, output)
    typer.echo(f"Wrote {len(records)} records to {output}")
    if stats:
        summary = summarize_attempts(records)
        typer.echo(format_summary(summary))


if __name__ == "__main__":
    app()
