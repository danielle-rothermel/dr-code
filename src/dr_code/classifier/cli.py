"""Typer integration for failure classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from dr_code.classifier.classify import (
    MAX_CONCURRENCY,
    ClassificationSummary,
    _classification_output_lock,
    _output_lock_path,
    _run_classification_locked,
    _staged_artifact_path,
    build_classification_experiment,
)
from dr_code.classifier.lane import SubscriptionLane
from dr_code.classifier.records import (
    ClassifierExperimentRecord,
    experiment_identity,
)
from dr_code.corpus.run_descriptor import RunDescriptor
from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.database import (
    DatabaseOwnershipError,
    ViewerDatabase,
    database_owner_lock_path,
)

DEFAULT_DATABASE = Path(".runs/dr-code-viewer.duckdb")
DEFAULT_TIMEOUT_SECONDS = 120.0


def register(app: typer.Typer) -> None:
    """Register the classifier command on the shared application."""

    @app.command("classify-failures")
    def classify_failures(  # noqa: PLR0913
        descriptor_path: Annotated[
            Path,
            typer.Argument(
                exists=True,
                file_okay=True,
                dir_okay=False,
                readable=True,
                help="Run descriptor JSON path.",
            ),
        ],
        provider: Annotated[
            str,
            typer.Option(help="Subscription provider identity."),
        ],
        model: Annotated[
            str,
            typer.Option(help="Provider model identity."),
        ],
        timeout: Annotated[
            float,
            typer.Option(
                min=0.001,
                help="Per-command timeout in seconds.",
            ),
        ] = DEFAULT_TIMEOUT_SECONDS,
        database: Annotated[
            Path,
            typer.Option(
                dir_okay=False,
                help="Viewer annotation database.",
            ),
        ] = DEFAULT_DATABASE,
        details: Annotated[
            Path | None,
            typer.Option(
                dir_okay=False,
                help="Canonical classification JSONL path.",
            ),
        ] = None,
        repeats: Annotated[
            int,
            typer.Option(min=1, max=25, help="Repeats per failure."),
        ] = 5,
        parse_limit: Annotated[
            int,
            typer.Option(
                min=0,
                help="Parse failures to select; 0 means uncapped.",
            ),
        ] = 300,
        test_limit: Annotated[
            int,
            typer.Option(
                min=0,
                help="Test failures to select; 0 means uncapped.",
            ),
        ] = 100,
        concurrency: Annotated[
            int,
            typer.Option(
                min=1,
                max=MAX_CONCURRENCY,
                help="Concurrent failure items.",
            ),
        ] = 4,
        force: Annotated[
            bool,
            typer.Option(help="Ignore exact matching resume records."),
        ] = False,
    ) -> None:
        """Classify parse and measured candidate-test failures."""
        try:
            lane = SubscriptionLane(
                provider=provider,
                model=model,
                timeout_seconds=timeout,
            )
            resolved_descriptor_path = descriptor_path.expanduser().resolve()
            descriptor = RunDescriptor.from_file(resolved_descriptor_path)
            database_path = database.expanduser().resolve()
            selected_parse_limit = _optional_limit(parse_limit)
            selected_test_limit = _optional_limit(test_limit)
            experiment = build_classification_experiment(
                descriptor,
                lane,
                repeats=repeats,
                parse_limit=selected_parse_limit,
                test_limit=selected_test_limit,
            )
            if details is not None and details.expanduser().is_symlink():
                raise ValueError(
                    "classification details path must not be a symbolic link"
                )
            details_path = (
                details.expanduser().resolve()
                if details is not None
                else _default_details_path(database_path, experiment)
            )
            _validate_classifier_paths(
                details_path=details_path,
                database_path=database_path,
                descriptor_path=resolved_descriptor_path,
                descriptor=descriptor,
            )
            with _classification_output_lock(details_path):
                with ViewerDatabase(database_path) as viewer_database:
                    analytics = ViewerAnalytics(
                        viewer_database,
                        (descriptor,),
                    )
                    with analytics.classification_serialization():
                        summary = _run_classification_locked(
                            analytics,
                            descriptor,
                            lane,
                            details_path=details_path,
                            repeats=repeats,
                            parse_limit=selected_parse_limit,
                            test_limit=selected_test_limit,
                            concurrency=concurrency,
                            force=force,
                        )
        except (DatabaseOwnershipError, OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(_summary_json(summary))


def _optional_limit(value: int) -> int | None:
    return value or None


def _default_details_path(
    database_path: Path,
    experiment: ClassifierExperimentRecord,
) -> Path:
    scope = experiment_identity(experiment)
    return database_path.parent / "failure-classifications" / f"{scope}.jsonl"


def _summary_json(summary: ClassificationSummary) -> str:
    return json.dumps(
        {
            "classified": summary.classified,
            "dataset_id": summary.dataset_id,
            "details_path": str(summary.details_path),
            "details_sha256": summary.details_sha256,
            "experiment_identity": summary.experiment_identity,
            "label_counts": summary.label_counts,
            "mean_agreement": summary.mean_agreement,
            "model": summary.model,
            "parse_selected": summary.parse_selected,
            "parse_total": summary.parse_total,
            "prompt_version": summary.prompt_version,
            "provider": summary.provider,
            "repeat_failures": summary.repeat_failures,
            "repeats": summary.repeats,
            "resumed": summary.resumed,
            "run_id": summary.run_id,
            "tasks_protected": summary.tasks_protected,
            "tasks_removed": summary.tasks_removed,
            "tasks_written": summary.tasks_written,
            "taxonomy_version": summary.taxonomy_version,
            "test_selected": summary.test_selected,
            "test_total": summary.test_total,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_classifier_paths(
    *,
    details_path: Path,
    database_path: Path,
    descriptor_path: Path,
    descriptor: RunDescriptor,
) -> None:
    stage_path = _staged_artifact_path(details_path)
    lock_path = _output_lock_path(details_path)
    mutable = {
        "classification details": details_path,
        "classification database": database_path,
        "classification database owner lock": database_owner_lock_path(
            database_path
        ),
        "classification staged artifact": stage_path,
        "classification output lock": lock_path,
    }
    items = tuple(mutable.items())
    for index, (left_label, left_path) in enumerate(items):
        for right_label, right_path in items[index + 1 :]:
            if _paths_collide(left_path, right_path):
                raise ValueError(
                    f"{left_label} path collides with {right_label} path"
                )

    artifacts: list[tuple[str, Path]] = [
        ("run descriptor", descriptor_path),
        ("corpus", descriptor.corpus_path),
        ("preprocessing manifest", descriptor.preprocessing_manifest_path),
        ("preprocessing results", descriptor.results_path),
        ("preprocessing candidates", descriptor.candidates_path),
        ("preprocessing step facts", descriptor.step_facts_path),
        ("preprocessing rejections", descriptor.rejections_path),
    ]
    for label, path in (
        ("evaluation manifest", descriptor.evaluation_manifest_path),
        ("candidate membership", descriptor.candidate_membership_path),
        ("candidate results", descriptor.candidate_results_path),
    ):
        if path is not None:
            artifacts.append((label, path))
    if descriptor.evaluation_root_path is not None:
        artifacts.append(
            (
                "evaluation generation pointer",
                descriptor.evaluation_root_path / "CURRENT.json",
            )
        )
    for mutable_label, mutable_path in mutable.items():
        for artifact_label, artifact_path in artifacts:
            if _paths_collide(mutable_path, artifact_path):
                raise ValueError(
                    f"{mutable_label} path collides with descriptor "
                    f"artifact {artifact_label}"
                )


def _paths_collide(left: Path, right: Path) -> bool:
    canonical_left = left.expanduser().resolve()
    canonical_right = right.expanduser().resolve()
    if canonical_left == canonical_right:
        return True
    try:
        return canonical_left.samefile(canonical_right)
    except FileNotFoundError:
        return False


__all__ = ("register",)
