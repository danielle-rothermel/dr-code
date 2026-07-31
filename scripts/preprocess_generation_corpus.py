"""Run exhaustive preprocessing over a generation-corpus Parquet file."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.corpus.preprocessing_run import run_preprocessing_corpus

app = typer.Typer(
    help="Preprocess a generation corpus into resumable Parquet artifacts.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    input_path: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Input Parquet with sample_id and nullable decoder_output.",
    ),
    output_root: Path = typer.Option(
        ...,
        "--output-root",
        file_okay=False,
        help="Directory containing immutable preprocessing runs.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Unique run identifier; generated when omitted.",
    ),
    batch_size: int = typer.Option(
        1_000,
        "--batch-size",
        min=1,
        help="Rows processed per bounded in-memory batch.",
    ),
    max_row_groups: int | None = typer.Option(
        None,
        "--max-row-groups",
        min=1,
        help="Stop after this many newly completed row groups.",
    ),
) -> None:
    """Write or resume one preprocessing run."""

    destination = run_preprocessing_corpus(
        input_path=input_path,
        output_root=output_root,
        run_id=run_id,
        batch_size=batch_size,
        max_row_groups=max_row_groups,
    )
    typer.echo(destination)


if __name__ == "__main__":
    app()
