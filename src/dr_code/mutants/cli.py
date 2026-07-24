"""Typer CLI for deterministic behavioral-mutant suite generation."""

from __future__ import annotations

from pathlib import Path

import typer

from dr_code.mutants.dataset import save_dataset
from dr_code.mutants.generate import generate_mutants
from dr_code.mutants.operators import ALL_FAMILIES, OperatorFamily, iter_sites
from dr_code.synthetic.humaneval_loader import load_humaneval_plus

ALL_KEYWORD = "all"

app = typer.Typer(
    name="mutants",
    help="Generate the seeded behavioral-mutant suite for HumanEval+.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Generate the seeded behavioral-mutant suite for HumanEval+."""


def _select_families(raw: str) -> tuple[OperatorFamily, ...]:
    if raw == ALL_KEYWORD:
        return ALL_FAMILIES
    selected: list[OperatorFamily] = []
    unknown: list[str] = []
    for token in raw.split(","):
        name = token.strip()
        if not name:
            continue
        try:
            selected.append(OperatorFamily(name))
        except ValueError:
            unknown.append(name)
    if unknown:
        known = ", ".join(f.value for f in ALL_FAMILIES)
        raise typer.BadParameter(
            f"unknown operator family/families: {', '.join(unknown)}. "
            f"Known: {known}",
            param_hint="--operators",
        )
    if not selected:
        raise typer.BadParameter(
            "provide at least one operator family or 'all'",
            param_hint="--operators",
        )
    return tuple(selected)


def _task_filter(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(token.strip() for token in raw.split(",") if token.strip())


@app.command()
def generate(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Directory to write mutants.jsonl + manifest.json "
        "(required unless --dry-run).",
    ),
    operators: str = typer.Option(
        ALL_KEYWORD,
        "--operators",
        help="Comma-separated operator families, or 'all'.",
    ),
    seeds: int = typer.Option(
        1, "--seeds", min=1, help="Seed indices 0..N-1 per (task, family)."
    ),
    tasks: str = typer.Option(
        "", "--tasks", help="Comma-separated task_ids, or empty for all 164."
    ),
    max_inputs_per_mutant: int = typer.Option(
        50,
        "--max-inputs",
        min=1,
        help="Deterministic head subsample of test inputs per mutant.",
    ),
    timeout_seconds: float = typer.Option(
        5.0, "--timeout", min=0.1, help="Per-execution subprocess timeout."
    ),
    use_snapshot: bool = typer.Option(
        True,
        "--snapshot/--hf",
        help="Load the offline snapshot (default) or the pinned HF revision.",
    ),
    compose_rename: bool = typer.Option(
        False,
        "--compose-rename",
        help="Also rename each mutant's entry point to target_fxn "
        "(behavior-preserving; optional).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List applicable sites per task/family; do not execute or write.",
    ),
) -> None:
    """Generate the mutant suite (or list sites with --dry-run)."""

    families = _select_families(operators)
    task_ids = _task_filter(tasks)

    if dry_run:
        _dry_run(
            families=families,
            task_ids=task_ids,
            use_snapshot=use_snapshot,
        )
        return

    if output_dir is None:
        raise typer.BadParameter(
            "--output-dir is required unless --dry-run",
            param_hint="--output-dir",
        )

    records, manifest = generate_mutants(
        families=families,
        seeds=seeds,
        max_inputs_per_mutant=max_inputs_per_mutant,
        timeout_seconds=timeout_seconds,
        task_filter=task_ids,
        compose_rename=compose_rename,
        prefer_snapshot=use_snapshot,
    )
    mutants_path, manifest_path = save_dataset(
        output_dir=output_dir, records=records, manifest=manifest
    )
    typer.echo(f"accepted mutants: {manifest.accepted_count}")
    for name, count in manifest.accepted_by_family:
        typer.echo(f"  {name}: {count}")
    typer.echo(f"config identity: {manifest.config_identity}")
    typer.echo(f"wrote {mutants_path}")
    typer.echo(f"wrote {manifest_path}")


def _dry_run(
    *,
    families: tuple[OperatorFamily, ...],
    task_ids: tuple[str, ...],
    use_snapshot: bool,
) -> None:
    tasks = load_humaneval_plus(prefer_snapshot=use_snapshot)
    wanted = set(task_ids)
    for task in tasks:
        if wanted and task.task_id not in wanted:
            continue
        source = task.full_source
        for family in families:
            sites = iter_sites(source, family)
            if not sites:
                continue
            typer.echo(f"{task.task_id} {family.value}: {len(sites)} site(s)")
            for site in sites:
                typer.echo(f"    #{site.node_path} {site.description}")


__all__ = ["app"]
