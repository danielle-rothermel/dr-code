"""Tests for the synthetic dataset builder."""

from __future__ import annotations

from pathlib import Path

from dr_code.synthetic import (
    RECIPES,
    RECIPES_BY_NAME,
    build_sample,
    build_dataset,
    load_dataset,
    save_dataset,
)
from dr_code.synthetic.humaneval_loader import HumanEvalPlusTask
from dr_code.synthetic.corruption_recipes import recipe_coordinate
from dr_code.synthetic.models import (
    RecipeCoordinate,
    SyntheticSample,
    SyntheticSampleCoordinate,
)

TASKS = (
    HumanEvalPlusTask(
        task_id="HumanEval/0",
        prompt="def add(a, b):\n",
        canonical_solution="    return a + b\n",
        entry_point="add",
        test="",
    ),
    HumanEvalPlusTask(
        task_id="HumanEval/1",
        prompt="def subtract(a, b):\n",
        canonical_solution="    return a - b\n",
        entry_point="subtract",
        test="",
    ),
)


def test_recipe_set_is_unique() -> None:
    names = [recipe.name for recipe in RECIPES]
    assert len(names) == len(set(names)), f"duplicate recipe names: {names}"
    assert set(names) == set(RECIPES_BY_NAME)


def test_dataset_size_matches_cross_product() -> None:
    samples = build_dataset(tasks=TASKS, seed=123)
    assert len(samples) == len(TASKS) * len(RECIPES)


def test_dataset_jsonl_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    save_dataset(build_dataset(tasks=TASKS, seed=8675309), first_path)
    save_dataset(build_dataset(tasks=TASKS, seed=8675309), second_path)

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(
        encoding="utf-8"
    )


def test_same_coordinate_produces_same_generated_sample() -> None:
    recipe = RECIPES_BY_NAME["kitchen_sink"]

    assert build_sample(TASKS[0], recipe, 7) == build_sample(
        TASKS[0], recipe, 7
    )


def test_structured_sample_coordinate_distinguishes_complete_inputs() -> None:
    version_zero = RecipeCoordinate(
        recipe_name="clean", version="0", corruptions=()
    )
    version_one = version_zero.model_copy(update={"version": "1"})
    coordinates = {
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            generation_seed=7,
            recipe=version_zero,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/1",
            generation_seed=7,
            recipe=version_zero,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            generation_seed=8,
            recipe=version_zero,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            generation_seed=7,
            recipe=version_one,
        ),
    }

    assert len(coordinates) == 4


def test_dataset_jsonl_roundtrip_preserves_boundary_schema(
    tmp_path: Path,
) -> None:
    samples = build_dataset(tasks=TASKS, seed=42)
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
        "coordinate",
        "ground_truth_source",
        "corrupted_source",
    }
    assert first_row.coordinate == SyntheticSampleCoordinate(
        humaneval_task_id="HumanEval/0",
        generation_seed=42,
        recipe=recipe_coordinate(RECIPES_BY_NAME["clean"]),
    )
