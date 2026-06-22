"""Stage 2 demo: single AttemptRecord parse walkthrough + Mongo inspect hints."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import typer

from dr_code.datasets.export import read_attempts
from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
    compute_sample_id,
)
from dr_code.parsing.adapter import parse_attempt
from dr_code.parsing.config import EXTRACTION_CONFIG, config_fingerprint
from dr_code.parsing.display import (
    format_eval_result_reference,
    format_parse_walkthrough,
)

app = typer.Typer(add_completion=False)

_DEFAULT_TASK_ID = "HumanEval/0"
_DEMO_POOL_EXPORT = Path("exports/demo/pool.parquet")
_FIXTURE_PARQUET = Path("tests/fixtures/pool/sample.parquet")
_POOL_SAMPLES = Path("../code-eval/tests/corpus/pool_samples.jsonl")
_DEFAULT_MONGODB_URL = "mongodb://localhost:27017/dr_queues"


def _section(title: str) -> None:
    typer.echo("")
    typer.echo(f"=== {title} ===")


def _resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    if _DEMO_POOL_EXPORT.is_file():
        return _DEMO_POOL_EXPORT
    return _FIXTURE_PARQUET


def _select_record(
    records: list[AttemptRecord],
    *,
    task_id: str,
    index: int,
) -> AttemptRecord:
    matches = [record for record in records if record.task_id == task_id]
    if not matches:
        msg = f"No records found for task_id={task_id!r} in input export"
        raise typer.BadParameter(msg)
    if index < 0 or index >= len(matches):
        msg = (
            f"--index {index} out of range for {len(matches)} matching row(s)"
        )
        raise typer.BadParameter(msg)
    return matches[index]


def _load_failure_record() -> AttemptRecord:
    if not _POOL_SAMPLES.is_file():
        msg = f"Failure fixture missing: {_POOL_SAMPLES}"
        raise typer.BadParameter(msg)
    for line in _POOL_SAMPLES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("expect_success") is False:
            task_id = str(row["task_id"])
            raw_output = str(row["raw_output"])
            return AttemptRecord(
                sample_id=compute_sample_id(task_id, raw_output),
                run_id=None,
                task_id=task_id,
                entry_point="has_close_elements",
                decoder_input="fixture",
                raw_output=raw_output,
                provenance=AttemptProvenance(source=AttemptSource.POOL),
            )
    msg = "No expect_success=false row found in pool_samples.jsonl"
    raise typer.BadParameter(msg)


def _print_mongo_commands(
    *,
    run_id: str,
    sample_id: str,
    mongodb_url: str,
    outcome_json: str,
) -> None:
    typer.echo("")
    typer.echo(
        "Reference eval_results document shape "
        "(from this demo's ParseOutcome):"
    )
    typer.echo(outcome_json)
    typer.echo("")
    typer.echo(
        "When the stage 2–3 pipeline is wired, inspect MongoDB with "
        f"run_id={run_id!r} and sample_id={sample_id!r}:"
    )
    typer.echo("")
    typer.echo("# Count parse-stage pipeline events for this run")
    typer.echo(
        f"mongosh {mongodb_url} \\\n"
        f"  --eval 'db.pipeline_events.countDocuments("
        f'{{run_id: "{run_id}", stage: "parse"}}'
        f")'"
    )
    typer.echo("")
    typer.echo("# Preview parse stage_output payload for this sample")
    typer.echo(
        f"mongosh {mongodb_url} \\\n"
        f"  --eval 'db.pipeline_events.find("
        f'{{run_id: "{run_id}", stage: "parse", '
        f'"payload.sample_id": "{sample_id}"}}'
        f").limit(1).pretty()'"
    )
    typer.echo("")
    typer.echo(
        "# Query projected eval result (future eval_results collection)"
    )
    typer.echo(
        f"mongosh {mongodb_url} \\\n"
        f"  --eval 'db.eval_results.findOne("
        f'{{run_id: "{run_id}", sample_id: "{sample_id}"}}'
        f")'"
    )
    typer.echo("")
    typer.echo(
        "Note: counts will be zero until the dr-queues pipeline runs with "
        "Mongo (docker compose up -d in dr-queues). Re-run these commands "
        "with the printed run_id and sample_id after that lands."
    )


@app.command()
def main(
    input_path: Path | None = typer.Option(
        None,
        "--input",
        help="AttemptRecord Parquet/JSONL export (default: demo pool or fixture)",
    ),
    task_id: str = typer.Option(
        _DEFAULT_TASK_ID,
        "--task-id",
        help="HumanEval task id to select from input export",
    ),
    index: int = typer.Option(
        0,
        "--index",
        help="Pick the Nth matching row for task_id",
    ),
    show_failure: bool = typer.Option(
        False,
        "--show-failure",
        help="Also parse a known failure row from pool_samples.jsonl",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run id substituted into Mongo inspect commands",
    ),
    mongodb_url: str = typer.Option(
        _DEFAULT_MONGODB_URL,
        "--mongodb-url",
        help="MongoDB URL for printed mongosh commands",
    ),
) -> None:
    """Walk one AttemptRecord through code-eval parsing."""
    resolved_input = _resolve_input_path(input_path)
    if not resolved_input.is_file():
        msg = f"Input export not found: {resolved_input}"
        raise typer.BadParameter(msg)

    demo_run_id = run_id or f"demo-{uuid4().hex[:8]}"
    records = read_attempts(resolved_input)
    record = _select_record(records, task_id=task_id, index=index)

    _section("Load attempt")
    typer.echo(f"Input: {resolved_input}")
    typer.echo(f"Selected row index={index} for task_id={task_id!r}")
    typer.echo(f"  sample_id: {record.sample_id}")
    typer.echo(f"  run_id: {record.run_id}")
    typer.echo(f"  task_id: {record.task_id}")
    typer.echo(f"  entry_point: {record.entry_point}")
    typer.echo(f"  source: {record.provenance.source.value}")
    typer.echo(f"  occurrence_count: {record.provenance.occurrence_count}")
    typer.echo("")
    typer.echo("raw_output:")
    typer.echo(record.raw_output)

    _section("Parse config")
    typer.echo("Using code-eval EXTRACTION_CONFIG (normalizers=())")
    typer.echo(f"config_fingerprint: {config_fingerprint()}")
    typer.echo(f"EXTRACTION_CONFIG: {EXTRACTION_CONFIG!r}")

    _section("Run parse")
    outcome = parse_attempt(record)
    if outcome.latency_ms is not None:
        typer.echo(f"latency_ms: {outcome.latency_ms:.2f}")

    _section("Results")
    typer.echo(format_parse_walkthrough(record, outcome))

    if show_failure:
        _section("Failure case (pool_samples expect_success=false)")
        failure_record = _load_failure_record()
        failure_outcome = parse_attempt(failure_record)
        typer.echo(format_parse_walkthrough(failure_record, failure_outcome))

    _section("MongoDB — inspect when pipeline is wired")
    effective_run_id = record.run_id or demo_run_id
    _print_mongo_commands(
        run_id=effective_run_id,
        sample_id=record.sample_id,
        mongodb_url=mongodb_url,
        outcome_json=format_eval_result_reference(outcome),
    )

    typer.echo("")
    typer.echo("Stage 2 demo completed successfully.")


if __name__ == "__main__":
    app()
