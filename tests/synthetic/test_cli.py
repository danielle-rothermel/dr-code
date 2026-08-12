from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.humaneval.sampling import (
    DEFAULT_HUMANEVAL_DATASET_NAME,
    DEFAULT_HUMANEVAL_HF_REVISION,
    HumanEvalRawRow,
    HumanEvalRawRowsSnapshot,
    HumanEvalRawRowsSnapshotHeader,
)
from dr_code.humaneval.task import HUMANEVAL_OVERRIDE_SET
from dr_code.synthetic.models import SyntheticSample


@pytest.fixture
def one_row_snapshot_path(tmp_path: Path) -> Path:
    snapshot = HumanEvalRawRowsSnapshot(
        header=HumanEvalRawRowsSnapshotHeader(
            schema_version=2,
            dataset_id=DEFAULT_HUMANEVAL_DATASET_NAME,
            hf_revision=DEFAULT_HUMANEVAL_HF_REVISION,
            override_set=HUMANEVAL_OVERRIDE_SET,
        ),
        rows=(
            HumanEvalRawRow(
                task_id="HumanEval/0",
                prompt="def add_one(x):\n",
                canonical_solution="    return x + 1\n",
                entry_point="add_one",
                test=(
                    "def check(candidate):\n"
                    "    inputs = [(1,)]\n"
                    "    results = [2]\n"
                    "    for inp, expected in zip(inputs, results):\n"
                    "        assertion(candidate(*inp), expected)\n"
                ),
            ),
        ),
    )
    snapshot_path = tmp_path / "humanevalplus_snapshot.json"
    snapshot_path.write_text(snapshot.model_dump_json(), encoding="utf-8")
    return snapshot_path


def test_cli_build_writes_requested_dataset(
    tmp_path: Path,
    one_row_snapshot_path: Path,
    run_python_module,
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
        "--snapshot",
        str(one_row_snapshot_path),
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
    assert [sample.coordinate.recipe.recipe_name for sample in samples] == [
        "clean"
    ]
    assert [sample.coordinate.recipe.version for sample in samples] == ["0"]


def test_cli_build_rejects_unknown_recipe(
    tmp_path: Path,
    one_row_snapshot_path: Path,
    run_python_module,
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
        "--snapshot",
        str(one_row_snapshot_path),
        "--output",
        str(output_path),
    )

    assert result.returncode != 0
    assert "unknown recipe(s): unknown" in result.stderr
    assert not output_path.exists()


def test_cli_build_rejects_more_tasks_than_snapshot_contains(
    tmp_path: Path,
    one_row_snapshot_path: Path,
    run_python_module,
) -> None:
    output_path = tmp_path / "synthetic.jsonl"

    result = run_python_module(
        "dr_code.synthetic",
        "build",
        "--recipes",
        "clean",
        "--tasks",
        "2",
        "--seed",
        "7",
        "--snapshot",
        str(one_row_snapshot_path),
        "--output",
        str(output_path),
    )

    assert result.returncode != 0
    normalized_stderr = " ".join(result.stderr.split())
    assert "requested task count exceeds available" in normalized_stderr
    assert "HumanEvalPlus tasks: 2 > 1" in normalized_stderr
    assert not output_path.exists()
