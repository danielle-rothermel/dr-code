from __future__ import annotations

from pathlib import Path

import pytest

from _paths import HUMANEVALPLUS_SNAPSHOT
import drc_synthetic.dataset_builder as dataset_builder
from drc_synthetic import (
    RECIPES,
    RECIPES_BY_NAME,
    InapplicableRecipeError,
    build_sample,
    build_dataset,
    load_dataset,
    save_dataset,
)
from drc_humaneval.plus_dataset import HumanEvalPlusTask
from drc_synthetic.corruption_recipes import recipe_coordinate
from drc_synthetic.models import (
    RecipeCoordinate,
    SyntheticSample,
    SyntheticSampleCoordinate,
)

TASK_ZERO_SOURCE_SHA256 = (
    "8f75a68646c879fd0cff5c02708c010fb1c01d67ef9930d4e419106feb7897aa"
)
TASK_ONE_SOURCE_SHA256 = (
    "58503bef65c62ed04279dcca0a154139c1e574aea5c321413eec60cab10e3e76"
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


def test_dataset_skips_inapplicable_recipe_rows() -> None:
    samples = build_dataset(tasks=TASKS, seed=123)

    assert len(samples) < len(TASKS) * len(RECIPES)
    assert all(
        not sample.coordinate.recipe.corruptions
        or sample.corrupted_source != sample.ground_truth_source
        for sample in samples
    )


def test_one_shot_recipe_generator_covers_applicable_task_cross_product() -> (
    None
):
    applicable_recipes = (
        RECIPES_BY_NAME["clean"],
        RECIPES_BY_NAME["fenced_tagged"],
    )
    samples = build_dataset(
        tasks=TASKS,
        recipes=(recipe for recipe in applicable_recipes),
        seed=123,
    )

    assert [
        (
            sample.coordinate.humaneval_task_id,
            sample.coordinate.recipe.recipe_name,
        )
        for sample in samples
    ] == [
        (task.task_id, recipe.name)
        for task in TASKS
        for recipe in applicable_recipes
    ]


@pytest.mark.parametrize(
    ("snapshot_path", "prefer_snapshot"),
    [(None, False), (Path("offline-snapshot.json"), True)],
)
def test_build_dataset_selects_requested_task_source(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_path: Path | None,
    prefer_snapshot: bool,
) -> None:
    calls: list[tuple[bool, Path | None]] = []

    def fake_load_humaneval_plus(
        *, prefer_snapshot: bool, snapshot_path: Path | None
    ) -> list[HumanEvalPlusTask]:
        calls.append((prefer_snapshot, snapshot_path))
        return []

    monkeypatch.setattr(
        dataset_builder, "load_humaneval_plus", fake_load_humaneval_plus
    )

    assert build_dataset(snapshot_path=snapshot_path) == []
    assert calls == [(prefer_snapshot, snapshot_path)]


def test_build_dataset_rejects_explicit_tasks_with_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("explicit tasks must not load another task source")

    monkeypatch.setattr(
        dataset_builder, "load_humaneval_plus", unexpected_load
    )

    with pytest.raises(ValueError):
        build_dataset(tasks=TASKS, snapshot_path=tmp_path / "snapshot.json")


def test_build_dataset_snapshot_path_is_keyword_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_load(*args: object, **kwargs: object) -> None:
        pytest.fail("explicit tasks must not load another task source")

    monkeypatch.setattr(
        dataset_builder, "load_humaneval_plus", unexpected_load
    )

    with pytest.raises(TypeError):
        build_dataset((), (), 0, Path("snapshot.json"))


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


def test_source_changes_produce_distinct_coordinates_and_sample_ids() -> None:
    recipe = RECIPES_BY_NAME["clean"]
    revised_task = TASKS[0].model_copy(
        update={"canonical_solution": "    return a * b\n"}
    )

    original = build_sample(TASKS[0], recipe, 7)
    revised = build_sample(revised_task, recipe, 7)

    assert original.ground_truth_source != revised.ground_truth_source
    assert original.coordinate != revised.coordinate
    assert original.sample_id != revised.sample_id


def test_build_sample_reports_inapplicable_recipe() -> None:
    recipe = RECIPES_BY_NAME["quote_style_swap"]

    with pytest.raises(
        InapplicableRecipeError,
        match=(
            r"synthetic recipe quote_style_swap@0 is not applicable to "
            r"HumanEval/0"
        ),
    ):
        build_sample(TASKS[0], recipe, 7)


def test_different_seeds_produce_different_corruption_witnesses() -> None:
    recipe = RECIPES_BY_NAME["kitchen_sink"]

    first = build_sample(TASKS[0], recipe, 7)
    second = build_sample(TASKS[0], recipe, 8)

    assert first.corrupted_source != second.corrupted_source


def test_structured_sample_coordinate_distinguishes_complete_inputs() -> None:
    version_zero = RecipeCoordinate(
        recipe_name="clean", version="0", corruptions=()
    )
    version_one = version_zero.model_copy(update={"version": "1"})
    coordinates = {
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            ground_truth_source_sha256=TASK_ZERO_SOURCE_SHA256,
            generation_seed=7,
            recipe=version_zero,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/1",
            ground_truth_source_sha256=TASK_ONE_SOURCE_SHA256,
            generation_seed=7,
            recipe=version_zero,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            ground_truth_source_sha256=TASK_ZERO_SOURCE_SHA256,
            generation_seed=8,
            recipe=version_zero,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            ground_truth_source_sha256=TASK_ZERO_SOURCE_SHA256,
            generation_seed=7,
            recipe=version_one,
        ),
        SyntheticSampleCoordinate(
            humaneval_task_id="HumanEval/0",
            ground_truth_source_sha256=TASK_ONE_SOURCE_SHA256,
            generation_seed=7,
            recipe=version_zero,
        ),
    }

    assert len(coordinates) == 5


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
        ground_truth_source_sha256=TASK_ZERO_SOURCE_SHA256,
        generation_seed=42,
        recipe=recipe_coordinate(RECIPES_BY_NAME["clean"]),
    )


def test_sample_id_layout_is_pinned() -> None:
    # The literal pins the persisted sample ID layout.
    coordinate = SyntheticSampleCoordinate(
        humaneval_task_id="HumanEval/0",
        ground_truth_source_sha256=TASK_ZERO_SOURCE_SHA256,
        generation_seed=1,
        recipe=recipe_coordinate(RECIPES_BY_NAME["clean"]),
    )

    assert SyntheticSample.make_id(coordinate) == (
        "HumanEval/0::"
        "8f75a68646c879fd0cff5c02708c010fb1c01d67ef9930d4e419106feb7897aa"
        "::clean@0::1"
    )


def test_sample_coordinate_json_layout_is_pinned() -> None:
    # The literal JSON pins persisted identity and RNG seed material.
    coordinate = SyntheticSampleCoordinate(
        humaneval_task_id="HumanEval/0",
        ground_truth_source_sha256=TASK_ZERO_SOURCE_SHA256,
        generation_seed=1,
        recipe=recipe_coordinate(RECIPES_BY_NAME["fenced_tagged"]),
    )

    assert coordinate.model_dump_json() == (
        '{"humaneval_task_id":"HumanEval/0",'
        '"ground_truth_source_sha256":'
        '"8f75a68646c879fd0cff5c02708c010fb1c01d67ef9930d4e419106feb7897aa",'
        '"generation_seed":1,'
        '"recipe":{"recipe_name":"fenced_tagged","version":"0",'
        '"corruptions":[{"registered_name":"add_code_fences","version":"0",'
        '"settings":[{"name":"language_tag","value":"python"}]}]}}'
    )


def test_canonical_unicode_and_quote_strata_contain_only_changed_rows() -> (
    None
):
    corpus_path = HUMANEVALPLUS_SNAPSHOT
    samples = build_dataset(
        recipes=(
            RECIPES_BY_NAME["unicode_fullwidth"],
            RECIPES_BY_NAME["quote_style_swap"],
        ),
        seed=0,
        snapshot_path=corpus_path,
    )
    counts = {
        recipe_name: sum(
            sample.coordinate.recipe.recipe_name == recipe_name
            for sample in samples
        )
        for recipe_name in ("unicode_fullwidth", "quote_style_swap")
    }

    assert counts == {"unicode_fullwidth": 164, "quote_style_swap": 60}
    assert all(
        sample.corrupted_source != sample.ground_truth_source
        for sample in samples
    )
