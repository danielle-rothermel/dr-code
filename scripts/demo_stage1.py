"""Stage 1 demo: HumanEval+ loader, pool import, fresh generation, spot check."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import typer

from dr_code.datasets.display import format_spot_check
from dr_code.datasets.export import read_attempts, write_attempts
from dr_code.datasets.humaneval_loader import get_task, load_humaneval_plus
from dr_code.datasets.pool_loader import (
    load_pool_dedup_with_parquet,
    load_pool_parquet,
)
from dr_code.datasets.stats import format_summary, summarize_attempts
from dr_code.generation.batch import generate_attempts
from dr_code.generation.profiles import default_profiles_path
from dr_code.generation.prompts import build_decoder_prompt
from dr_code.generation.run_config import GenerationRunConfig
from dr_code.models.attempts import AttemptRecord

app = typer.Typer(add_completion=False)

_FIXTURE_PARQUET = Path("tests/fixtures/pool/sample.parquet")
_FIXTURE_DEDUP = Path("tests/fixtures/pool/human_eval-0-decode-dedup.jsonl")
_DEMO_POOL_EXPORT = Path("exports/demo/pool.parquet")
_DEMO_FRESH_EXPORT = Path("exports/demo/fresh.parquet")
_DEFAULT_PROFILE = "openrouter/google/gemini-3.1-flash-lite/off/v1"
_DEFAULT_TASK_ID = "HumanEval/0"


def _section(title: str) -> None:
    typer.echo("")
    typer.echo(f"=== {title} ===")


@app.command()
def main(
    pool_parquet: Path | None = typer.Option(
        None,
        "--pool-parquet",
        help="Override pool Parquet path",
    ),
    pool_dedup: Path | None = typer.Option(
        None,
        "--pool-dedup",
        help="Override dedup JSONL path",
    ),
    skip_live: bool = typer.Option(
        False,
        "--skip-live",
        help="Skip live dr-providers generation (offline smoke)",
    ),
    profile: str = typer.Option(
        _DEFAULT_PROFILE,
        "--profile",
        help="OpenRouter profile for fresh generation",
    ),
    profiles_path: Path = typer.Option(
        default_profiles_path(),
        "--profiles-path",
        help="Path to OpenRouter profiles YAML",
    ),
    task_id: str = typer.Option(
        _DEFAULT_TASK_ID,
        "--task-id",
        help="HumanEval task id for fresh generation and spot check",
    ),
) -> None:
    """Run stage 1 verification: pool import, fresh collection, spot check."""
    _section("HumanEval+ loader")
    tasks = load_humaneval_plus(prefer_snapshot=True)
    typer.echo(f"Loaded {len(tasks)} tasks from snapshot")
    for task_num in (0, 20, 100):
        sample = get_task(f"HumanEval/{task_num}", prefer_snapshot=True)
        typer.echo(
            f"  HumanEval/{task_num}: entry_point={sample.entry_point!r}, "
            f"prompt_bytes={len(sample.prompt.encode())}, "
            f"test_bytes={len(sample.test.encode())}"
        )

    _section("Decoder prompt preview")
    spot_task = get_task(task_id, prefer_snapshot=True)
    prompt = build_decoder_prompt(spot_task)
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
    pool_records = parquet_records + dedup_records

    _section("Fresh collection")
    run_id = f"demo-{uuid4().hex[:8]}"
    if skip_live:
        typer.echo(
            "Skipping live generation (--skip-live). "
            "Using stub fresh_stub record."
        )
        fresh_records = [
            AttemptRecord.stub_for_fresh(
                spot_task,
                decoder_input=spot_task.prompt,
                raw_output=(
                    "def has_close_elements(numbers, threshold):\n    pass\n"
                ),
                run_id=run_id,
                model="demo/model",
                profile_id=profile,
            )
        ]
    else:
        typer.echo(
            f"Generating fresh_stub for {task_id} via profile {profile!r} "
            f"(run_id={run_id})"
        )
        config = GenerationRunConfig(
            run_id=run_id,
            profile_id=profile,
            profiles_path=profiles_path,
            task_ids=[task_id],
        )
        fresh_records = generate_attempts(config)
        typer.echo(f"Generated {len(fresh_records)} fresh_stub record(s)")

    _section("Export")
    write_attempts(pool_records, _DEMO_POOL_EXPORT)
    write_attempts(fresh_records, _DEMO_FRESH_EXPORT)
    pool_roundtrip = read_attempts(_DEMO_POOL_EXPORT)
    fresh_roundtrip = read_attempts(_DEMO_FRESH_EXPORT)
    if len(pool_roundtrip) != len(pool_records):
        msg = (
            f"Pool round-trip count mismatch: "
            f"{len(pool_roundtrip)} != {len(pool_records)}"
        )
        raise RuntimeError(msg)
    if len(fresh_roundtrip) != len(fresh_records):
        msg = (
            f"Fresh round-trip count mismatch: "
            f"{len(fresh_roundtrip)} != {len(fresh_records)}"
        )
        raise RuntimeError(msg)
    typer.echo(
        f"Pool export OK: {len(pool_roundtrip)} records at {_DEMO_POOL_EXPORT}"
    )
    typer.echo(
        f"Fresh export OK: {len(fresh_roundtrip)} records at {_DEMO_FRESH_EXPORT}"
    )

    _section("Stats — pool")
    typer.echo(
        format_summary(summarize_attempts(pool_records, sample_count=2))
    )

    _section("Stats — fresh_stub")
    typer.echo(
        format_summary(summarize_attempts(fresh_records, sample_count=2))
    )

    _section("Side-by-side spot check")
    typer.echo(
        format_spot_check(
            pool_records,
            fresh_records,
            task_id=task_id,
        )
    )

    typer.echo("")
    typer.echo("Stage 1 demo completed successfully.")


if __name__ == "__main__":
    app()
