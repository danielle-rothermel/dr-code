"""Typer subcommand: classify a run's preprocessing failures via an LLM lane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import duckdb
import typer

from dr_code.classifier.classify import MAX_CONCURRENCY, run_classification
from dr_code.classifier.lane import DEFAULT_LANE, PiLane, known_lanes


def _default_detail_path(database: Path) -> Path:
    return database.parent / "classifier-details.jsonl"


def register(app: typer.Typer) -> None:
    """Attach the ``classify-failures`` command to a Typer app."""

    @app.command("classify-failures")
    def classify_failures(  # noqa: PLR0913 - CLI surface
        run: Annotated[
            str,
            typer.Option(
                "--run",
                help="Run descriptor as LABEL=/path/to/run.json.",
            ),
        ],
        database: Annotated[
            Path,
            typer.Option(
                "--database",
                dir_okay=False,
                help="DuckDB catalog and annotation database.",
            ),
        ] = Path(".runs/dr-code-viewer.duckdb"),
        details: Annotated[
            Path | None,
            typer.Option(
                "--details",
                dir_okay=False,
                help="Per-example JSONL artifact path "
                "(default: next to the database).",
            ),
        ] = None,
        lane: Annotated[
            str,
            typer.Option(
                "--lane",
                help="Subscription lane: " + ", ".join(known_lanes()),
            ),
        ] = DEFAULT_LANE,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Override the lane's model id."),
        ] = None,
        repeats: Annotated[
            int,
            typer.Option("--repeats", min=1, max=25, help="Repeats per item."),
        ] = 5,
        parse_limit: Annotated[
            int,
            typer.Option(
                "--parse-limit",
                min=1,
                help="Cap on parse failures classified.",
            ),
        ] = 300,
        test_limit: Annotated[
            int,
            typer.Option(
                "--test-limit", min=1, help="Cap on test failures classified."
            ),
        ] = 100,
        include_tests: Annotated[
            bool,
            typer.Option(
                "--include-tests/--no-tests",
                help="Also classify test failures when evaluation exists.",
            ),
        ] = True,
        concurrency: Annotated[
            int,
            typer.Option(
                "--concurrency",
                min=1,
                max=MAX_CONCURRENCY,
                help="Concurrent lane calls (capped at 4).",
            ),
        ] = MAX_CONCURRENCY,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help="Reclassify items already labelled at this taxonomy.",
            ),
        ] = False,
    ) -> None:
        """Classify preprocessing failures and persist machine annotations."""
        label, separator, raw_path = run.partition("=")
        label = label.strip()
        raw_path = raw_path.strip()
        if not separator or not label or not raw_path:
            raise typer.BadParameter(
                "expected LABEL=/path/to/run.json", param_hint="--run"
            )
        descriptor_path = Path(raw_path).expanduser()
        if not descriptor_path.is_file():
            raise typer.BadParameter(
                f"descriptor does not exist: {descriptor_path}",
                param_hint="--run",
            )
        database_path = database.expanduser().resolve()
        detail_path = (
            details.expanduser().resolve()
            if details is not None
            else _default_detail_path(database_path)
        )
        try:
            pi_lane = PiLane.for_lane(lane, model=model)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--lane") from error

        # Import here to keep the CLI import light and match viewer wiring.
        from dr_code.viewer.cli import build_service

        try:
            service = build_service(
                [(label, descriptor_path.resolve())], database_path
            )
        except (OSError, ValueError, duckdb.Error) as error:
            raise typer.BadParameter(str(error), param_hint="--run") from error

        descriptor = service.list_runs()[0]
        run_descriptor = service._runs[descriptor.run_id]  # noqa: SLF001

        summary = run_classification(
            service,
            run_descriptor,
            pi_lane,
            detail_path=detail_path,
            repeats=repeats,
            parse_limit=parse_limit,
            test_limit=test_limit,
            include_tests=include_tests,
            concurrency=concurrency,
            force=force,
        )
        typer.echo(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "lane": summary.lane,
                    "model": summary.model,
                    "taxonomy_version": summary.taxonomy_version,
                    "repeats": summary.repeats,
                    "parse_total": summary.parse_total,
                    "parse_classified": summary.parse_classified,
                    "test_total": summary.test_total,
                    "test_classified": summary.test_classified,
                    "skipped": summary.skipped,
                    "typed_failures": summary.typed_failures,
                    "mean_agreement": summary.mean_agreement,
                    "min_agreement": summary.min_agreement,
                    "label_distribution": summary.label_distribution,
                    "tasks_written": summary.tasks_written,
                    "human_collisions_skipped": (
                        summary.human_collisions_skipped
                    ),
                    "detail_path": str(summary.detail_path),
                },
                indent=2,
                sort_keys=True,
            )
        )


__all__ = ("register",)
