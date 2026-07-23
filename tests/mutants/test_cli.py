"""CLI tests: dry-run listing, generate, and one-command regeneration."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dr_code.mutants.cli import app
from dr_code.mutants.dataset import dataset_filenames

runner = CliRunner()


def test_dry_run_lists_applicable_sites() -> None:
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
    assert "HumanEval/0 comparison_flip" in result.output
    assert "site(s)" in result.output


def test_dry_run_needs_no_output_dir() -> None:
    result = runner.invoke(
        app, ["generate", "--dry-run", "--tasks", "HumanEval/0"]
    )
    assert result.exit_code == 0, result.output


def test_generate_requires_output_dir_without_dry_run() -> None:
    result = runner.invoke(
        app, ["generate", "--tasks", "HumanEval/0", "--seeds", "1"]
    )
    assert result.exit_code != 0
    assert "output-dir" in result.output


def test_unknown_operator_is_rejected() -> None:
    result = runner.invoke(
        app, ["generate", "--dry-run", "--operators", "not_a_family"]
    )
    assert result.exit_code != 0
    assert "unknown operator" in result.output


def test_generate_writes_artifacts_and_regenerates_identically(
    tmp_path: Path,
) -> None:
    args = [
        "generate",
        "--tasks",
        "HumanEval/0,HumanEval/7",
        "--operators",
        "comparison_flip,range_inclusivity",
        "--seeds",
        "1",
        "--max-inputs",
        "15",
    ]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    r1 = runner.invoke(app, [*args, "--output-dir", str(out_a)])
    r2 = runner.invoke(app, [*args, "--output-dir", str(out_b)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert "accepted mutants:" in r1.output
    assert "config identity:" in r1.output

    mutants_name, manifest_name = dataset_filenames()
    assert (out_a / mutants_name).read_bytes() == (
        out_b / mutants_name
    ).read_bytes()
    assert (out_a / manifest_name).read_bytes() == (
        out_b / manifest_name
    ).read_bytes()
