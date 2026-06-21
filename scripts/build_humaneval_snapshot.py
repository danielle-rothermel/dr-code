"""Build and save the HumanEval+ offline snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from dr_code.datasets.humaneval_loader import load_humaneval_plus, save_snapshot

app = typer.Typer(add_completion=False)


@app.command()
def main(
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override snapshot path (default: tests/corpus/...)",
    ),
) -> None:
    """Download HumanEval+ from Hugging Face and write offline snapshot."""
    tasks = load_humaneval_plus(prefer_snapshot=False)
    if output is None:
        path = save_snapshot(tasks)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                [task.model_dump() for task in tasks],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        path = output
    size_bytes = path.stat().st_size
    typer.echo(f"Wrote {len(tasks)} tasks to {path} ({size_bytes} bytes)")


if __name__ == "__main__":
    app()
