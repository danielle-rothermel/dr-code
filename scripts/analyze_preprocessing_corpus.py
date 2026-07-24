"""Create compact deterministic deliverables from a preprocessing run."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.corpus.preprocessing_analysis import analyze_preprocessing_corpus

app = typer.Typer(
    help=(
        "Analyze immutable preprocessing Parquets into compact, "
        "hash-authenticated artifacts."
    ),
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    dataset_id: str = typer.Option(
        ..., "--dataset-id", help="Canonical dataset identity."
    ),
    corpus_path: Path = typer.Option(
        ..., "--corpus", exists=True, dir_okay=False
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", exists=True, file_okay=False
    ),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    candidate_evaluation: Path | None = typer.Option(
        None,
        "--candidate-evaluation",
        exists=True,
        file_okay=False,
        help="Optional complete candidate-evaluation directory.",
    ),
) -> None:
    """Validate and summarize one completed preprocessing run."""
    artifacts = analyze_preprocessing_corpus(
        dataset_id=dataset_id,
        corpus_path=corpus_path,
        run_dir=run_dir,
        output_dir=output_dir,
        candidate_evaluation=candidate_evaluation,
    )
    typer.echo(artifacts.output_dir)


if __name__ == "__main__":
    app()
