"""Reproduce bounded escaped-newline extraction counts without payload output."""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
from dataclasses import dataclass
from typing import Annotated

import typer

from dr_code.humaneval.code_parsing import (
    BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
    LEGACY_PARSER_PROFILE_VERSION,
    PARSER_PROFILE_VERSION,
    extract_code_with_profile,
    resolve_parser_profile,
)

MAX_ROWS = 500
MODEL = "openai/gpt-5.4-nano"
EXTRACTION_ERROR = "no code candidates extracted"


@dataclass(frozen=True)
class ReplayRow:
    prediction_id: str
    decoded_generation: str


def _query(limit: int) -> str:
    return f"""
SELECT
    prediction_id,
    replace(encode(convert_to(decoded_generation, 'UTF8'), 'base64'),
            chr(10), ''),
    current_setting('transaction_read_only')
FROM public.dr_dspy_encdec_eval_predictions
WHERE decoder_model = '{MODEL}'
  AND extraction_error = '{EXTRACTION_ERROR}'
  AND decoded_generation IS NOT NULL
ORDER BY prediction_id
LIMIT {limit}
"""


def _load_rows(database_url: str, limit: int) -> list[ReplayRow]:
    environment = os.environ.copy()
    pg_options = environment.get("PGOPTIONS", "").strip()
    environment["PGOPTIONS"] = (
        f"{pg_options} -c default_transaction_read_only=on".strip()
    )
    completed = subprocess.run(
        [
            "psql",
            "--dbname",
            database_url,
            "--no-psqlrc",
            "--set",
            "ON_ERROR_STOP=1",
            "--no-align",
            "--tuples-only",
            "--field-separator",
            "\t",
            "--command",
            _query(limit),
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    if completed.returncode != 0:
        typer.echo(
            "read-only replay query failed; payload output suppressed",
            err=True,
        )
        raise typer.Exit(code=1)

    rows: list[ReplayRow] = []
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or fields[2] != "on":
            typer.echo(
                "replay did not prove a read-only transaction", err=True
            )
            raise typer.Exit(code=1)
        try:
            decoded_generation = base64.b64decode(
                fields[1], validate=True
            ).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            typer.echo(
                "replay row decoding failed; payload output suppressed",
                err=True,
            )
            raise typer.Exit(code=1) from None
        rows.append(
            ReplayRow(
                prediction_id=fields[0],
                decoded_generation=decoded_generation,
            )
        )
    return rows


def _classify(rows: list[ReplayRow], parser_version: str) -> dict[str, int]:
    profile = resolve_parser_profile(
        parser_profile_id=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE_ID,
        parser_version=parser_version,
    )
    counts = {"compilable": 0, "noncompilable": 0, "no_candidates": 0}
    for row in rows:
        result = extract_code_with_profile(
            row.decoded_generation,
            profile=profile,
        )
        if result.succeeded:
            counts["compilable"] += 1
        elif result.candidate_count:
            counts["noncompilable"] += 1
        else:
            counts["no_candidates"] += 1
    return counts


def main(
    database_url: Annotated[
        str | None,
        typer.Option(
            envvar="DATABASE_URL",
            help="Postgres URL; read from DATABASE_URL by default.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(min=1, max=MAX_ROWS, help="Bounded replay row count."),
    ] = MAX_ROWS,
) -> None:
    """Print selection and compile counts; never print stored payloads."""
    if database_url is None:
        typer.echo("DATABASE_URL is required", err=True)
        raise typer.Exit(code=2)

    rows = _load_rows(database_url, limit)
    selection_hash = hashlib.sha256(
        "\n".join(row.prediction_id for row in rows).encode()
    ).hexdigest()
    typer.echo("transaction_read_only=on")
    typer.echo(f"selected_rows={len(rows)}")
    typer.echo(f"selection_id_sha256={selection_hash}")
    for version in (LEGACY_PARSER_PROFILE_VERSION, PARSER_PROFILE_VERSION):
        counts = _classify(rows, version)
        for outcome, count in counts.items():
            typer.echo(f"parser_{version}_{outcome}={count}")


if __name__ == "__main__":
    typer.run(main)
