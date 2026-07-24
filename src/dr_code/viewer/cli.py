"""Command-line entry point for the local preprocessing viewer."""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import duckdb
import typer
import uvicorn

from dr_code.viewer.app import ViewerService, create_app

DEFAULT_DATABASE = Path(".runs/dr-code-viewer.duckdb")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

app = typer.Typer(
    name="dr-code",
    help="dr-code development tools.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Run dr-code development tools."""


def _loopback_host(value: str) -> str:
    """Reject hosts that would expose this unauthenticated local tool."""
    if value.lower() == "localhost":
        return value
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise typer.BadParameter(
            "must be a loopback IP address or 'localhost'"
        ) from error
    if not address.is_loopback:
        raise typer.BadParameter(
            "must be a loopback IP address or 'localhost'"
        )
    return value


def _parse_run_options(values: Sequence[str]) -> list[Path]:
    registrations: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        raw_path = value.strip()
        if not raw_path:
            raise typer.BadParameter(
                "expected /path/to/run.json",
                param_hint="--run",
            )
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise typer.BadParameter(
                f"descriptor does not exist or is not a file: {path}",
                param_hint="--run",
            )
        resolved = path.resolve()
        if resolved in seen:
            raise typer.BadParameter(
                f"duplicate run descriptor: {resolved}",
                param_hint="--run",
            )
        seen.add(resolved)
        registrations.append(resolved)
    return registrations


def build_service(
    registrations: Sequence[Path], database_path: Path
) -> ViewerService:
    """Construct the core service at the CLI integration boundary."""
    from dr_code.viewer.analytics import ViewerAnalytics
    from dr_code.viewer.database import ViewerDatabase
    from dr_code.viewer.domain import RunDescriptor

    descriptors = [RunDescriptor.from_file(path) for path in registrations]
    database = ViewerDatabase(database_path)
    return ViewerAnalytics(database, descriptors)


@app.command()
def viewer(
    runs: Annotated[
        list[str],
        typer.Option(
            "--run",
            help="JSON run descriptor path. Repeat for additional runs.",
        ),
    ],
    database: Annotated[
        Path,
        typer.Option(
            "--database",
            dir_okay=False,
            help="DuckDB catalog and annotation database.",
        ),
    ] = DEFAULT_DATABASE,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            callback=_loopback_host,
            help="Loopback address to bind.",
        ),
    ] = DEFAULT_HOST,
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65535, help="Port to bind."),
    ] = DEFAULT_PORT,
) -> None:
    """Serve registered preprocessing runs in one local process."""
    registrations = _parse_run_options(runs)
    if not registrations:
        raise typer.BadParameter(
            "provide at least one named run descriptor",
            param_hint="--run",
        )
    database_path = database.expanduser().resolve()
    try:
        service = build_service(registrations, database_path)
    except (OSError, ValueError, duckdb.Error) as error:
        raise typer.BadParameter(str(error), param_hint="--run") from error
    application = create_app(service, allowed_host=host)
    # Annotation writes require a single process; do not expose a worker option.
    uvicorn.run(application, host=host, port=port, workers=1)


if __name__ == "__main__":
    app()
