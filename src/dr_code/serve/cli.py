"""CLI for the serve facade: run the local server or dump its OpenAPI spec.

Facades bind to localhost only — the host is intentionally not an
option.
"""

import json

import typer

LOCALHOST = "127.0.0.1"
DEFAULT_PORT = 8321

app = typer.Typer(help="dr-code serve facade", no_args_is_help=True)


@app.command()
def serve(port: int = typer.Option(DEFAULT_PORT, help="Localhost port")) -> None:
    """Run the explain facade on localhost."""
    import uvicorn

    from dr_code.serve.app import create_app

    typer.echo(f"dr-code serve listening on http://{LOCALHOST}:{port}")
    uvicorn.run(create_app(), host=LOCALHOST, port=port, log_level="info")


@app.command()
def openapi() -> None:
    """Print the OpenAPI schema (input for generated TS clients)."""
    from dr_code.serve.app import create_app

    typer.echo(json.dumps(create_app().openapi(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
