"""Create compact deterministic deliverables from a preprocessing run."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.corpus.preprocessing_analysis import analyze_preprocessing_corpus

app = typer.Typer(
    help=(
        "Analyze preprocessing Parquets into compact viewer-ready artifacts. "
        "After evaluate_preprocessing_candidates.py completes, pass its "
        "manifest and both Parquet exports to include test-outcome joins."
    ),
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def run(
    corpus_path: Path = typer.Option(
        ..., "--corpus", exists=True, dir_okay=False
    ),
    run_dir: Path = typer.Option(
        ..., "--run-dir", exists=True, file_okay=False
    ),
    output_dir: Path = typer.Option(..., "--output-dir", file_okay=False),
    candidate_membership_path: Path | None = typer.Option(
        None,
        "--candidate-membership",
        exists=True,
        dir_okay=False,
        help="candidate_membership.parquet from candidate evaluation.",
    ),
    candidate_results_path: Path | None = typer.Option(
        None,
        "--candidate-results",
        exists=True,
        dir_okay=False,
        help="candidate_results.parquet from the same evaluation run.",
    ),
    candidate_evaluation_manifest_path: Path | None = typer.Option(
        None,
        "--candidate-evaluation-manifest",
        exists=True,
        dir_okay=False,
        help="candidate_evaluation_manifest.json from the same evaluation run.",
    ),
) -> None:
    """Validate and summarize one completed preprocessing run."""
    artifacts = analyze_preprocessing_corpus(
        corpus_path=corpus_path,
        run_dir=run_dir,
        output_dir=output_dir,
        candidate_membership_path=candidate_membership_path,
        candidate_results_path=candidate_results_path,
        candidate_evaluation_manifest_path=(
            candidate_evaluation_manifest_path
        ),
    )
    typer.echo(artifacts.output_dir)


if __name__ == "__main__":
    app()
