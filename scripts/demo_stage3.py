"""Stage 3 demo: single-sample stages 1–3 walkthrough + Mongo inspect hints."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import typer

from dr_code.datasets.export import read_attempts
from dr_code.datasets.humaneval_loader import get_task
from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
    compute_sample_id,
)
from dr_code.models.outcomes import ParseOutcome, TestOutcome
from dr_code.parsing.adapter import parse_attempt
from dr_code.parsing.config import config_fingerprint
from dr_code.parsing.display import format_parse_walkthrough
from dr_code.testing.adapter import test_parsed_sample
from dr_code.testing.bridge import (
    load_test_cases,
    supports_function_call_tests,
)
from dr_code.testing.config import default_timeout_seconds
from dr_code.testing.display import (
    format_eval_result_reference,
    format_outcome_banner,
    format_test_walkthrough,
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
        "Reference eval_results document shape (from this demo's TestOutcome):"
    )
    typer.echo(outcome_json)
    typer.echo("")
    typer.echo(
        "When the stage 2–3 pipeline is wired, inspect MongoDB with "
        f"run_id={run_id!r} and sample_id={sample_id!r}:"
    )
    typer.echo("")
    typer.echo("# Count test-stage pipeline events for this run")
    typer.echo(
        f"mongosh {mongodb_url} \\\n"
        f"  --eval 'db.pipeline_events.countDocuments("
        f'{{run_id: "{run_id}", stage: "test"}}'
        f")'"
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
        "Mongo (docker compose up -d in dr-queues)."
    )


def _run_walkthrough(
    record: AttemptRecord,
    *,
    label: str | None = None,
) -> tuple[ParseOutcome, TestOutcome]:
    if label is not None:
        _section(label)

    task = get_task(record.task_id, prefer_snapshot=True)
    _section("Stage 1 — Task context")
    typer.echo(f"  task_id: {task.task_id}")
    typer.echo(f"  entry_point: {task.entry_point}")
    typer.echo(f"  prompt_bytes: {len(task.prompt.encode())}")
    typer.echo(f"  test_bytes: {len(task.test.encode())}")
    if supports_function_call_tests(task):
        typer.echo(f"  test_case_count: {len(load_test_cases(task))}")
    else:
        typer.echo("  test_case_count: unsupported (inputs_ref_func)")

    _section("Stage 1 — Load attempt")
    typer.echo(f"  sample_id: {record.sample_id}")
    typer.echo(f"  run_id: {record.run_id}")
    typer.echo(f"  source: {record.provenance.source.value}")
    typer.echo(f"  occurrence_count: {record.provenance.occurrence_count}")
    typer.echo("")
    typer.echo("raw_output:")
    typer.echo(record.raw_output)

    _section("Stage 2 — Parse config")
    typer.echo("Using code-eval EXTRACTION_CONFIG (normalizers=())")
    typer.echo(f"config_fingerprint: {config_fingerprint()}")

    _section("Stage 2 — Run parse")
    parse_outcome = parse_attempt(record)
    if parse_outcome.latency_ms is not None:
        typer.echo(f"latency_ms: {parse_outcome.latency_ms:.2f}")

    _section("Stage 2 — Parse results")
    typer.echo(format_parse_walkthrough(record, parse_outcome))

    _section("Stage 3 — Run tests")
    typer.echo("One local fork worker per sample.")
    typer.echo(f"timeout_seconds: {default_timeout_seconds()}")
    if not parse_outcome.parse_success:
        typer.echo("Parse failed — test stage will skip execution.")

    test_outcome = test_parsed_sample(record, parse_outcome, task=task)

    _section("Stage 3 — Test results")
    typer.echo(format_outcome_banner(test_outcome))
    typer.echo("")
    typer.echo(format_test_walkthrough(record, parse_outcome, test_outcome))

    return parse_outcome, test_outcome


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
        help="Also walk through a known parse-failure row",
    ),
    show_canonical: bool = typer.Option(
        False,
        "--show-canonical",
        help="Also run canonical solution through the test stage",
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
    """Walk one AttemptRecord through stages 1–3."""
    resolved_input = _resolve_input_path(input_path)
    if not resolved_input.is_file():
        msg = f"Input export not found: {resolved_input}"
        raise typer.BadParameter(msg)

    demo_run_id = run_id or f"demo-{uuid4().hex[:8]}"
    records = read_attempts(resolved_input)
    record = _select_record(records, task_id=task_id, index=index)

    _section("Input selection")
    typer.echo(f"Input: {resolved_input}")
    typer.echo(f"Selected row index={index} for task_id={task_id!r}")

    _parse_outcome, test_outcome = _run_walkthrough(record)

    if show_failure:
        failure_record = _load_failure_record()
        _run_walkthrough(
            failure_record,
            label="Failure case (pool_samples expect_success=false)",
        )

    if show_canonical:
        task = get_task(task_id, prefer_snapshot=True)
        canonical_record = AttemptRecord(
            sample_id=f"canonical-{task.task_id.replace('/', '-')}",
            run_id=demo_run_id,
            task_id=task.task_id,
            decoder_input="canonical",
            raw_output=task.prompt + task.canonical_solution,
            provenance=AttemptProvenance(source=AttemptSource.POOL),
        )
        _run_walkthrough(
            canonical_record,
            label="Canonical solution baseline",
        )

    _section("MongoDB — inspect when pipeline is wired")
    effective_run_id = record.run_id or demo_run_id
    _print_mongo_commands(
        run_id=effective_run_id,
        sample_id=record.sample_id,
        mongodb_url=mongodb_url,
        outcome_json=format_eval_result_reference(test_outcome),
    )

    typer.echo("")
    typer.echo("Stage 3 demo completed successfully.")


if __name__ == "__main__":
    app()
