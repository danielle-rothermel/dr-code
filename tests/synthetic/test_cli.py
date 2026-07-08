"""Smoke tests for the synthetic CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

from dr_code.synthetic.models import SyntheticSample


def test_module_import_smoke() -> None:
    import dr_code.synthetic as synthetic

    assert synthetic.RECIPES
    assert synthetic.RECIPES_BY_NAME["clean"].transforms == ()


def test_cli_build_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic.jsonl"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
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
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )

    samples = [
        SyntheticSample.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "Wrote 1 samples" in result.stdout
    assert len(samples) == 1
    assert samples[0].recipe_name == "clean"
