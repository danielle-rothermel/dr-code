"""Evaluate candidates from a completed preprocessing corpus run."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.corpus.candidate_evaluation import (
    evaluate_preprocessing_candidates,
)

app = typer.Typer(
    help="Evaluate preprocessing candidates against pinned HumanEval+.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    preprocessing_run: Path = typer.Option(
        ...,
        "--preprocessing-run",
        exists=True,
        file_okay=False,
        readable=True,
    ),
    corpus: Path = typer.Option(
        ...,
        "--corpus",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Original corpus Parquet with sample_id and task_id.",
    ),
    output: Path = typer.Option(..., "--output", file_okay=False),
    snapshot: Path = typer.Option(
        ...,
        "--snapshot",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Explicit pinned HumanEval+ raw-row snapshot.",
    ),
    max_workers: int = typer.Option(4, "--max-workers", min=1),
    max_infrastructure_retries: int = typer.Option(
        2,
        "--max-infrastructure-retries",
        min=0,
        help="Retries after transient execution infrastructure failures.",
    ),
    reuse_results_from: list[Path] | None = typer.Option(
        None,
        "--reuse-results-from",
        exists=True,
        file_okay=False,
        readable=True,
        help="Compatible completed evaluation directory; repeatable.",
    ),
) -> None:
    """Run or resume evaluation from explicit immutable inputs."""

    artifacts = evaluate_preprocessing_candidates(
        preprocessing_run=preprocessing_run,
        corpus_path=corpus,
        output_dir=output,
        snapshot_path=snapshot,
        max_workers=max_workers,
        max_infrastructure_retries=max_infrastructure_retries,
        reuse_results_from=reuse_results_from or (),
    )
    typer.echo(artifacts.output_dir)


if __name__ == "__main__":
    app()
