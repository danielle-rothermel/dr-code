"""Functional tests for the synthetic CLI."""

from __future__ import annotations

from pathlib import Path

from dr_code.synthetic.models import SyntheticSample


def test_cli_build_writes_requested_dataset(
    tmp_path: Path, run_python_module
) -> None:
    output_path = tmp_path / "synthetic.jsonl"

    result = run_python_module(
        "dr_code.synthetic",
        "build",
        "--recipes",
        "clean",
        "--tasks",
        "1",
        "--seed",
        "7",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "Wrote 1 samples" in result.stdout
    samples = [
        SyntheticSample.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [sample.recipe_name for sample in samples] == ["clean"]


def test_cli_build_rejects_unknown_recipe(
    tmp_path: Path, run_python_module
) -> None:
    output_path = tmp_path / "synthetic.jsonl"

    result = run_python_module(
        "dr_code.synthetic",
        "build",
        "--recipes",
        "unknown",
        "--tasks",
        "1",
        "--seed",
        "7",
        "--output",
        str(output_path),
    )

    assert result.returncode != 0
    assert "unknown recipe(s): unknown" in result.stderr
    assert not output_path.exists()
