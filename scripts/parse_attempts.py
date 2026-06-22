"""Batch parse AttemptRecord exports to ParseOutcome JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from dr_code.datasets.export import read_attempts
from dr_code.parsing.adapter import parse_attempt

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_path: Path = typer.Option(
        ...,
        "--input",
        help="Parquet or JSONL AttemptRecord export",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output",
        help="JSONL ParseOutcome output path",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Parse at most N records",
    ),
) -> None:
    """Parse an AttemptRecord export and write ParseOutcome JSONL."""
    records = read_attempts(input_path)
    if limit is not None:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            outcome = parse_attempt(record)
            outcomes.append(outcome)
            handle.write(outcome.model_dump_json())
            handle.write("\n")

    success_count = sum(1 for outcome in outcomes if outcome.parse_success)
    fail_count = len(outcomes) - success_count
    latencies = [
        outcome.latency_ms
        for outcome in outcomes
        if outcome.latency_ms is not None
    ]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    typer.echo(f"Parsed {len(outcomes)} record(s) from {input_path}")
    typer.echo(f"  parse_success: {success_count}")
    typer.echo(f"  parse_failed: {fail_count}")
    typer.echo(f"  mean_latency_ms: {mean_latency:.2f}")
    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
