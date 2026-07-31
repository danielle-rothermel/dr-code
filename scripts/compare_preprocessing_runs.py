"""Export an identity-level comparison of two immutable preprocessing runs."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.corpus.preprocessing_comparison import compare_preprocessing_runs


app = typer.Typer(
    help=(
        "Compare immutable before/after preprocessing bundles over one corpus "
        "without running evaluation."
    ),
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    dataset_id: str = typer.Option(
        ..., "--dataset-id", help="Canonical dataset identity."
    ),
    corpus: Path = typer.Option(
        ...,
        "--corpus",
        exists=True,
        dir_okay=False,
        readable=True,
        help="The exact corpus Parquet used by both preprocessing runs.",
    ),
    before_run: Path = typer.Option(
        ...,
        "--before-run",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    after_run: Path = typer.Option(
        ...,
        "--after-run",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        help="A new append-only destination; existing paths are refused.",
    ),
    before_evaluation: Path | None = typer.Option(
        None,
        "--before-evaluation",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    after_evaluation: Path | None = typer.Option(
        None,
        "--after-evaluation",
        exists=True,
        file_okay=False,
        readable=True,
    ),
) -> None:
    """Write deterministic Parquet identity rows and a reconciled summary."""

    artifacts = compare_preprocessing_runs(
        dataset_id=dataset_id,
        corpus_path=corpus,
        before_run=before_run,
        after_run=after_run,
        output_dir=output_dir,
        before_evaluation=before_evaluation,
        after_evaluation=after_evaluation,
    )
    typer.echo(artifacts.output_dir)


if __name__ == "__main__":
    app()
