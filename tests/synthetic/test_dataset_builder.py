"""Tests for the synthetic dataset builder."""

from __future__ import annotations

from pathlib import Path

from dr_code.synthetic.models import SyntheticSample
from dr_code.synthetic import (
    RECIPES,
    RECIPES_BY_NAME,
    build_dataset,
    load_dataset,
    load_humaneval_plus,
    save_dataset,
)


def test_recipe_set_is_unique() -> None:
    names = [recipe.name for recipe in RECIPES]
    assert len(names) == len(set(names)), f"duplicate recipe names: {names}"
    assert set(names) == set(RECIPES_BY_NAME)


def test_dataset_size_matches_cross_product() -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)[:2]
    samples = build_dataset(tasks=tasks, seed=123)
    assert len(samples) == len(tasks) * len(RECIPES)


def test_dataset_jsonl_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)[:3]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    save_dataset(build_dataset(tasks=tasks, seed=8675309), first_path)
    save_dataset(build_dataset(tasks=tasks, seed=8675309), second_path)

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(
        encoding="utf-8"
    )


def test_dataset_jsonl_roundtrip_uses_rescued_schema(tmp_path: Path) -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)[:2]
    samples = build_dataset(tasks=tasks, seed=42)
    output_path = tmp_path / "dataset.jsonl"

    count = save_dataset(samples, output_path)
    loaded = load_dataset(output_path)
    first_row = SyntheticSample.model_validate_json(
        output_path.read_text(encoding="utf-8").splitlines()[0]
    )

    assert count == len(samples)
    assert loaded == samples
    assert set(first_row.model_dump()) == {
        "sample_id",
        "humaneval_task_id",
        "recipe_name",
        "ground_truth_source",
        "corrupted_source",
    }
