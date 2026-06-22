"""Batch test AttemptRecord exports joined with ParseOutcome JSONL."""

from __future__ import annotations

import traceback
from collections import Counter
from pathlib import Path

import typer

from dr_code.datasets.export import read_attempts
from dr_code.models.attempts import AttemptRecord
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.testing.adapter import missing_parse_outcome, test_parsed_sample
from dr_code.testing.config import default_timeout_seconds

app = typer.Typer(add_completion=False)


def _load_parse_outcomes(path: Path) -> dict[str, ParseOutcome]:
    by_sample_id: dict[str, ParseOutcome] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        outcome = ParseOutcome.model_validate_json(line)
        by_sample_id[outcome.sample_id] = outcome
    return by_sample_id


def _run_one(
    record: AttemptRecord,
    parse_by_sample_id: dict[str, ParseOutcome],
) -> TestOutcome:
    parse_outcome = parse_by_sample_id.get(record.sample_id)
    if parse_outcome is None:
        return missing_parse_outcome(record)
    try:
        return test_parsed_sample(record, parse_outcome)
    except Exception as exc:
        return TestOutcome(
            sample_id=record.sample_id,
            run_id=parse_outcome.run_id or record.run_id,
            task_id=record.task_id,
            parse_success=parse_outcome.parse_success,
            outcome_kind="internal_error",
            skipped=False,
            tests_ran=False,
            internal_error=(
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ),
        )


def _summary_line(outcome: TestOutcome) -> str:
    if outcome.outcome_kind == "tested":
        rate = outcome.test_pass_rate
        rate_text = f"{rate:.2f}" if rate is not None else "n/a"
        return f"outcome_kind=tested pass_rate={rate_text}"
    return f"outcome_kind={outcome.outcome_kind}"


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
    output_path: Path = typer.Option(
        ...,
        "--output",
        help="JSONL TestOutcome output path",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Test at most N records",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop on first internal_error (debug only)",
    ),
) -> None:
    """Run test adapter over joined attempt + parse exports."""
    records = read_attempts(attempts_path)
    if limit is not None:
        records = records[:limit]
    parse_by_sample_id = _load_parse_outcomes(parse_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    tested_pass = 0
    tested_fail = 0

    with output_path.open("w", encoding="utf-8") as handle:
        total = len(records)
        for index, record in enumerate(records, start=1):
            outcome = _run_one(record, parse_by_sample_id)
            handle.write(outcome.model_dump_json())
            handle.write("\n")
            handle.flush()

            counts[outcome.outcome_kind] += 1
            if outcome.outcome_kind == "tested":
                if outcome.all_tests_passed:
                    tested_pass += 1
                else:
                    tested_fail += 1

            typer.echo(
                f"[{index}/{total}] sample_id={record.sample_id} "
                f"{_summary_line(outcome)}",
                err=True,
            )

            if fail_fast and outcome.outcome_kind == "internal_error":
                typer.echo("Stopping early (--fail-fast)", err=True)
                break

    typer.echo(f"Tested {total} record(s) from {attempts_path}")
    typer.echo(f"  skipped: {counts['skipped']}")
    typer.echo(f"  tested_pass: {tested_pass}")
    typer.echo(f"  tested_fail: {tested_fail}")
    typer.echo(f"  infra_error: {counts['infra_error']}")
    typer.echo(f"  internal_error: {counts['internal_error']}")
    typer.echo(f"  timeout_seconds: {default_timeout_seconds()}")
    typer.echo(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
