"""Stage 1 demo: HumanEval+ loader, pool import, export, and stats."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.datasets.export import read_attempts, write_attempts
from dr_code.datasets.humaneval_loader import get_task, load_humaneval_plus
from dr_code.datasets.pool_loader import (
    load_pool_dedup_with_parquet,
    load_pool_parquet,
)
from dr_code.datasets.stats import format_summary, summarize_attempts
from dr_code.generation.prompts import build_decoder_prompt
from dr_code.models.attempts import AttemptRecord

app = typer.Typer(add_completion=False)

_FIXTURE_PARQUET = Path("tests/fixtures/pool/sample.parquet")
_FIXTURE_DEDUP = Path("tests/fixtures/pool/human_eval-0-decode-dedup.jsonl")
_DEMO_EXPORT = Path("exports/demo/roundtrip.parquet")


def _section(title: str) -> None:
    typer.echo("")
    typer.echo(f"=== {title} ===")


@app.command()
def main(
    pool_parquet: Path | None = typer.Option(
        None,
        "--pool-parquet",
        help="Override pool Parquet path (default: tests/fixtures/pool/sample.parquet)",
    ),
    pool_dedup: Path | None = typer.Option(
        None,
        "--pool-dedup",
        help="Override dedup JSONL path",
    ),
) -> None:
    """Run stage 1 smoke demo with sampled outputs and stats."""
    _section("HumanEval+ loader")
    tasks = load_humaneval_plus(prefer_snapshot=True)
    typer.echo(f"Loaded {len(tasks)} tasks from snapshot")
    for task_num in (0, 20, 100):
        task = get_task(f"HumanEval/{task_num}", prefer_snapshot=True)
        typer.echo(
            f"  HumanEval/{task_num}: entry_point={task.entry_point!r}, "
            f"prompt_bytes={len(task.prompt.encode())}, "
            f"test_bytes={len(task.test.encode())}"
        )

    _section("Decoder prompt preview")
    task0 = get_task("HumanEval/0", prefer_snapshot=True)
    prompt = build_decoder_prompt(task0)
    preview = prompt if len(prompt) <= 500 else prompt[:497] + "..."
    typer.echo(preview)

    _section("Pool import")
    parquet_path = pool_parquet or _FIXTURE_PARQUET
    dedup_path = pool_dedup or _FIXTURE_DEDUP
    parquet_records = load_pool_parquet(parquet_path)
    dedup_records = load_pool_dedup_with_parquet(
        dedup_path,
        parquet_path,
    )
    typer.echo(f"Parquet fixture rows: {len(parquet_records)}")
    typer.echo(f"Dedup fixture rows: {len(dedup_records)}")
    records = parquet_records + dedup_records

    _section("Export round-trip")
    write_attempts(records, _DEMO_EXPORT)
    roundtrip = read_attempts(_DEMO_EXPORT)
    if len(roundtrip) != len(records):
        msg = (
            f"Round-trip count mismatch: {len(roundtrip)} != {len(records)}"
        )
        raise RuntimeError(msg)
    typer.echo(f"Round-trip OK: {len(roundtrip)} records at {_DEMO_EXPORT}")

    _section("Stats")
    summary = summarize_attempts(records, sample_count=3)
    typer.echo(format_summary(summary))

    _section("Stub fresh record")
    stub = AttemptRecord.stub_for_fresh(
        task0,
        decoder_input=task0.prompt,
        raw_output="def has_close_elements(numbers, threshold):\n    pass\n",
        run_id="demo-run",
        model="demo/model",
    )
    typer.echo(f"  sample_id: {stub.sample_id}")
    typer.echo(f"  source: {stub.provenance.source.value}")
    typer.echo(f"  run_id: {stub.run_id}")
    typer.echo(f"  task_id: {stub.task_id}")

    typer.echo("")
    typer.echo("Stage 1 demo completed successfully.")


if __name__ == "__main__":
    app()
