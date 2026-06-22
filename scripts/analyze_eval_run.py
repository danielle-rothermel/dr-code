"""Offline analysis CLI for stage 4 eval exports."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.analysis.aggregate import build_aggregates, build_summary
from dr_code.analysis.export import export_analysis
from dr_code.analysis.join import (
    enrich_eval_run,
    load_attempts,
    load_parse_outcomes,
    load_test_outcomes,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    attempts_path: Path = typer.Option(
        ...,
        "--attempts",
        help="Parquet or JSONL AttemptRecord export",
    ),
    parse_path: Path = typer.Option(
        ...,
        "--parse",
        help="JSONL ParseOutcome export",
    ),
    test_path: Path = typer.Option(
        ...,
        "--test",
        help="JSONL TestOutcome export",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        help="Directory for enriched Parquet, summary JSON, and aggregates",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Analyze at most N attempt records",
    ),
) -> None:
    """Join attempt, parse, and test exports; write analysis artifacts."""
    attempts = load_attempts(attempts_path)
    if limit is not None:
        attempts = attempts[:limit]

    parse_by_sample_id = load_parse_outcomes(parse_path)
    test_by_sample_id = load_test_outcomes(test_path)

    enriched, join_report = enrich_eval_run(
        attempts,
        parse_by_sample_id,
        test_by_sample_id,
    )
    summary = build_summary(enriched, join_report)
    aggregates = build_aggregates(enriched)
    artifacts = export_analysis(enriched, summary, aggregates, output_dir)

    typer.echo(f"Analyzed {join_report.attempt_count} attempt(s)")
    typer.echo(f"  outcome_kind_counts: {summary['outcome_kind_counts']}")
    typer.echo(f"  correctness_pass_rate: {summary['correctness_pass_rate']}")
    typer.echo(
        "  correctness_pass_rate_weighted: "
        f"{summary['correctness_pass_rate_weighted']}"
    )
    join_failures = summary["join_failures"]
    typer.echo(f"  missing_test: {join_failures['missing_test_count']}")
    typer.echo(f"Wrote {artifacts.enriched_path}")
    typer.echo(f"Wrote {artifacts.summary_path}")


if __name__ == "__main__":
    app()
