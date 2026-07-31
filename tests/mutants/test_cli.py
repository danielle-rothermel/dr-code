"""Typer command contracts for dry-run and generation."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dr_code.mutants import provenance as provenance_module
from dr_code.synthetic import humaneval_loader as humaneval_loader_module
from dr_code.humaneval.task import HUMAN_EVAL_OVERRIDES
from dr_code.mutants.cli import app
from dr_code.mutants.dataset import load_dataset
from dr_code.mutants.operators import ALL_FAMILIES, iter_sites
from dr_code.mutants.provenance import resolve_canonical_suite
from dr_code.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    HumanEvalSource,
)

runner = CliRunner()


def test_dry_run_lists_stable_site_addresses_without_output_dir() -> None:
    result = runner.invoke(
        app,
        [
            "generate",
            "--dry-run",
            "--tasks",
            "HumanEval/0",
            "--operators",
            "comparison_flip",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "HumanEval/0 comparison_flip: 1 site(s)" in result.output
    assert "#" in result.output
    assert ":" in result.output


def test_generation_requires_output_directory() -> None:
    result = runner.invoke(
        app,
        ["generate", "--tasks", "HumanEval/0"],
    )

    assert result.exit_code == 2


def test_explicit_hf_dry_run_never_reads_packaged_snapshot(
    monkeypatch,
) -> None:
    task = HumanEvalPlusTask(
        task_id="HumanEval/fixture",
        prompt="def f(x):\n",
        canonical_solution="    return x < 1\n",
        entry_point="f",
        test="",
    )
    monkeypatch.setattr(
        humaneval_loader_module,
        "_load_from_hf",
        lambda: [task],
    )
    monkeypatch.setattr(
        humaneval_loader_module,
        "packaged_snapshot_bytes",
        lambda: (_ for _ in ()).throw(
            AssertionError("packaged snapshot must not be consulted")
        ),
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "--hf",
            "--dry-run",
            "--tasks",
            "HumanEval/fixture",
            "--operators",
            "comparison_flip",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "HumanEval/fixture comparison_flip: 1 site(s)" in result.output


def test_humaneval_32_dry_run_matches_generation_canonical_suite(
    monkeypatch,
) -> None:
    raw_task = next(
        task
        for task in humaneval_loader_module.load_humaneval_plus(
            source=HumanEvalSource.SNAPSHOT
        )
        if task.task_id == "HumanEval/32"
    )
    monkeypatch.setattr(
        provenance_module,
        "load_humaneval_plus",
        lambda source: [raw_task],
    )
    canonical_task = resolve_canonical_suite(
        task_ids=(raw_task.task_id,),
        max_inputs=10,
        source=HumanEvalSource.SNAPSHOT,
    )[0]
    original_apply = provenance_module.apply_human_eval_override
    applied_to: list[str] = []

    def apply_once(row, overrides):
        applied_to.append(str(row["canonical_solution"]))
        return original_apply(row, overrides)

    monkeypatch.setattr(
        provenance_module,
        "apply_human_eval_override",
        apply_once,
    )

    result = runner.invoke(
        app,
        [
            "generate",
            "--dry-run",
            "--tasks",
            raw_task.task_id,
            "--max-inputs",
            "10",
        ],
    )

    expected_lines: list[str] = []
    for family in ALL_FAMILIES:
        sites = iter_sites(canonical_task.canonical_full_source, family)
        if not sites:
            continue
        expected_lines.append(
            f"{raw_task.task_id} {family.value}: {len(sites)} site(s)"
        )
        expected_lines.extend(
            f"  #{site.node_path}:{site.target_index} {site.description}"
            for site in sites
        )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == expected_lines
    assert applied_to == [raw_task.canonical_solution]
    assert (
        applied_to[0]
        != HUMAN_EVAL_OVERRIDES[raw_task.task_id].canonical_solution
    )


def test_cli_rejects_unknown_or_duplicate_selection() -> None:
    unknown_operator = runner.invoke(
        app,
        ["generate", "--dry-run", "--operators", "unknown"],
    )
    duplicate_task = runner.invoke(
        app,
        [
            "generate",
            "--dry-run",
            "--tasks",
            "HumanEval/0,HumanEval/0",
        ],
    )
    unknown_task = runner.invoke(
        app,
        [
            "generate",
            "--dry-run",
            "--tasks",
            "HumanEval/not-real",
        ],
    )

    assert unknown_operator.exit_code != 0
    assert "unknown operator family" in unknown_operator.output
    assert duplicate_task.exit_code != 0
    assert "must not repeat" in duplicate_task.output
    assert unknown_task.exit_code != 0
    assert "unknown HumanEval+ task" in unknown_task.output


def test_cli_generates_loadable_authenticated_dataset(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "mutants"
    result = runner.invoke(
        app,
        [
            "generate",
            "--output-dir",
            str(destination),
            "--tasks",
            "HumanEval/0",
            "--operators",
            "comparison_flip",
            "--seeds",
            "1",
            "--max-inputs",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "accepted mutants: 1" in result.output
    assert "config identity:" in result.output
    assert "dataset identity:" in result.output
    dataset_identity = next(
        line.removeprefix("dataset identity: ")
        for line in result.output.splitlines()
        if line.startswith("dataset identity: ")
    )
    loaded = load_dataset(
        destination,
        expected_dataset_identity=dataset_identity,
        max_manifest_bytes=128 * 1024,
        max_records_bytes=128 * 1024,
    )
    assert len(loaded.records) == 1
    assert loaded.records[0].task_id == "HumanEval/0"


def test_cli_generation_does_not_clobber_existing_dataset(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "mutants"
    args = [
        "generate",
        "--output-dir",
        str(destination),
        "--tasks",
        "HumanEval/2",
        "--operators",
        "comparison_flip",
        "--seeds",
        "1",
        "--max-inputs",
        "1",
    ]
    first = runner.invoke(app, args)
    before = (destination / "manifest.json").read_bytes()
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "already exists" in second.output
    assert (destination / "manifest.json").read_bytes() == before
