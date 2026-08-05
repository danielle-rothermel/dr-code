"""Typer CLI for synthetic corruption-corpus generation."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.synthetic.corruption_recipes import (
    RECIPES,
    RECIPES_BY_NAME,
    Recipe,
)
from dr_code.synthetic.dataset_builder import build_dataset, save_dataset
from dr_code.synthetic.humaneval_loader import load_humaneval_plus

ALL_RECIPES = "all"

app = typer.Typer(
    name="synthetic",
    help="Build synthetic corruption-corpus JSONL artifacts.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Build synthetic corruption-corpus JSONL artifacts."""


def _select_recipes(recipe_names: str) -> tuple[Recipe, ...]:
    if recipe_names == ALL_RECIPES:
        return RECIPES
    selected: list[Recipe] = []
    unknown: list[str] = []
    for raw_name in recipe_names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        recipe = RECIPES_BY_NAME.get(name)
        if recipe is None:
            unknown.append(name)
        else:
            selected.append(recipe)
    if unknown:
        known = ", ".join(sorted(RECIPES_BY_NAME))
        raise typer.BadParameter(
            f"unknown recipe(s): {', '.join(unknown)}. Known recipes: {known}",
            param_hint="--recipes",
        )
    if not selected:
        raise typer.BadParameter(
            "provide at least one recipe name or 'all'",
            param_hint="--recipes",
        )
    return tuple(selected)


@app.command()
def build(
    recipes: str = typer.Option(
        ...,
        "--recipes",
        help="Comma-separated recipe names, or 'all'.",
    ),
    tasks: int = typer.Option(
        ...,
        "--tasks",
        min=1,
        help="Number of HumanEvalPlus snapshot tasks to include.",
    ),
    snapshot: Path = typer.Option(
        ...,
        "--snapshot",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="HumanEvalPlus JSON snapshot to draw tasks from.",
    ),
    seed: int = typer.Option(
        ...,
        "--seed",
        help="Dataset seed mixed with each task and recipe name.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        dir_okay=False,
        writable=True,
        resolve_path=True,
        help="JSONL output path.",
    ),
) -> None:
    """Build a synthetic corruption-corpus JSONL artifact."""
    selected_recipes = _select_recipes(recipes)
    selected_tasks = load_humaneval_plus(
        prefer_snapshot=True, snapshot_path=snapshot
    )[:tasks]
    samples = build_dataset(
        tasks=selected_tasks, recipes=selected_recipes, seed=seed
    )
    count = save_dataset(samples, output)
    typer.echo(f"Wrote {count} samples to {output}")
