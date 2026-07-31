"""Command line interface for behavioral-mutant generation."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.mutants.dataset import publish_dataset
from dr_code.mutants.generate import generate_mutants
from dr_code.mutants.operators import (
    ALL_FAMILIES,
    OperatorFamily,
    iter_sites,
)
from dr_code.mutants.provenance import resolve_canonical_suite
from dr_code.synthetic.humaneval_loader import (
    HumanEvalSource,
)

app = typer.Typer(
    name="mutants",
    help="Generate deterministic behavioral mutants for HumanEval+.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Generate deterministic behavioral mutants for HumanEval+."""


def _families(raw: str) -> tuple[OperatorFamily, ...]:
    if raw.strip() == "all":
        return ALL_FAMILIES
    names = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not names:
        raise typer.BadParameter(
            "provide at least one operator family or 'all'",
            param_hint="--operators",
        )
    if len(set(names)) != len(names):
        raise typer.BadParameter(
            "operator families must not repeat",
            param_hint="--operators",
        )
    try:
        selected = tuple(OperatorFamily(name) for name in names)
    except ValueError as exc:
        known = ", ".join(family.value for family in ALL_FAMILIES)
        raise typer.BadParameter(
            f"unknown operator family; expected one of: {known}",
            param_hint="--operators",
        ) from exc
    return tuple(family for family in ALL_FAMILIES if family in selected)


def _tasks(raw: str) -> tuple[str, ...]:
    selected = tuple(
        value.strip() for value in raw.split(",") if value.strip()
    )
    if len(set(selected)) != len(selected):
        raise typer.BadParameter(
            "task ids must not repeat",
            param_hint="--tasks",
        )
    return selected


@app.command()
def generate(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="New directory for mutants.jsonl and manifest.json.",
    ),
    operators: str = typer.Option(
        "all",
        "--operators",
        help="Comma-separated operator families, or 'all'.",
    ),
    tasks: str = typer.Option(
        "",
        "--tasks",
        help="Comma-separated HumanEval+ task ids; empty selects all.",
    ),
    seeds: int = typer.Option(
        1,
        "--seeds",
        min=1,
        help="Seed coordinates per task and operator.",
    ),
    max_inputs: int = typer.Option(
        50,
        "--max-inputs",
        min=1,
        help="Maximum canonical test inputs evaluated per task.",
    ),
    timeout_seconds: float = typer.Option(
        5.0,
        "--timeout",
        min=0.1,
        help="Wall-clock limit for each program execution.",
    ),
    use_snapshot: bool = typer.Option(
        True,
        "--snapshot/--hf",
        help="Use the repository snapshot or pinned remote revision.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List stable mutation sites without execution or writes.",
    ),
) -> None:
    """Generate and atomically publish a behavioral-mutant dataset."""

    selected_families = _families(operators)
    selected_tasks = _tasks(tasks)
    dataset_source = (
        HumanEvalSource.SNAPSHOT if use_snapshot else HumanEvalSource.HF
    )
    if dry_run:
        try:
            _print_sites(
                families=selected_families,
                task_ids=selected_tasks,
                max_inputs=max_inputs,
                dataset_source=dataset_source,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        return
    if output_dir is None:
        raise typer.BadParameter(
            "--output-dir is required unless --dry-run",
            param_hint="--output-dir",
        )
    try:
        generated = generate_mutants(
            families=selected_families,
            seeds=seeds,
            max_inputs_per_mutant=max_inputs,
            timeout_seconds=timeout_seconds,
            task_ids=selected_tasks or None,
            dataset_source=dataset_source,
        )
        artifacts = publish_dataset(
            output_dir=output_dir,
            generated=generated,
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"accepted mutants: {artifacts.manifest.accepted_count}")
    for family_count in artifacts.manifest.accepted_by_family:
        typer.echo(
            f"  {family_count.operator_family.value}: {family_count.count}"
        )
    typer.echo(f"config identity: {artifacts.manifest.config_identity}")
    typer.echo(f"dataset identity: {artifacts.manifest.dataset_identity}")
    typer.echo(f"wrote {artifacts.records_path}")
    typer.echo(f"wrote {artifacts.manifest_path}")


def _print_sites(
    *,
    families: tuple[OperatorFamily, ...],
    task_ids: tuple[str, ...],
    max_inputs: int,
    dataset_source: HumanEvalSource,
) -> None:
    tasks = resolve_canonical_suite(
        task_ids=task_ids or None,
        max_inputs=max_inputs,
        source=dataset_source,
    )
    for task in tasks:
        source = task.canonical_full_source
        for family in families:
            sites = iter_sites(source, family)
            if not sites:
                continue
            typer.echo(f"{task.task_id} {family.value}: {len(sites)} site(s)")
            for site in sites:
                typer.echo(
                    f"  #{site.node_path}:{site.target_index} "
                    f"{site.description}"
                )


__all__ = ("app",)
