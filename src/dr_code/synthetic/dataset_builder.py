"""Build the synthetic test corpus."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from dr_code.synthetic.models import SyntheticSample, SyntheticSampleCoordinate
from dr_code.synthetic.corruption_recipes import (
    RECIPES,
    Recipe,
    apply_recipe,
    recipe_coordinate,
)
from dr_code.code_transforms import strip_docstrings
from dr_code.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    load_humaneval_plus,
)


def build_sample(
    task: HumanEvalPlusTask,
    recipe: Recipe,
    seed: int,
) -> SyntheticSample:
    """Build a single synthetic sample for one (task, recipe) pair."""
    ground_truth = strip_docstrings(task.full_source)
    coordinate = SyntheticSampleCoordinate(
        humaneval_task_id=task.task_id,
        generation_seed=seed,
        recipe=recipe_coordinate(recipe),
    )
    rng = random.Random(coordinate.model_dump_json())
    corrupted = apply_recipe(recipe, ground_truth, rng)
    return SyntheticSample(
        sample_id=SyntheticSample.make_id(coordinate),
        coordinate=coordinate,
        ground_truth_source=ground_truth,
        corrupted_source=corrupted.corrupted_source,
    )


def iter_dataset(
    tasks: Iterable[HumanEvalPlusTask],
    recipes: Iterable[Recipe] = RECIPES,
    seed: int = 0,
) -> Iterator[SyntheticSample]:
    """Yield one `SyntheticSample` per (task, recipe) pair."""
    recipes_list = list(recipes)
    for task in tasks:
        for recipe in recipes_list:
            yield build_sample(task, recipe, seed)


def build_dataset(
    tasks: Iterable[HumanEvalPlusTask] | None = None,
    recipes: Iterable[Recipe] = RECIPES,
    seed: int = 0,
    prefer_snapshot: bool = True,
    snapshot_path: Path | None = None,
) -> list[SyntheticSample]:
    """Build the full dataset list (in-memory).

    If `tasks` is None, load HumanEvalPlus through the explicit
    `prefer_snapshot` source choice; `snapshot_path` is then required when
    `prefer_snapshot` is True.
    """
    if tasks is None:
        tasks_iter: Iterable[HumanEvalPlusTask] = load_humaneval_plus(
            prefer_snapshot=prefer_snapshot, snapshot_path=snapshot_path
        )
    else:
        tasks_iter = tasks
    return list(iter_dataset(tasks_iter, recipes, seed))


def save_dataset(samples: Iterable[SyntheticSample], path: Path) -> int:
    """Serialize samples as JSONL to `path`. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(sample.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def load_dataset(path: Path) -> list[SyntheticSample]:
    """Read a dataset JSONL artifact back into `SyntheticSample` objects."""
    samples: list[SyntheticSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            samples.append(SyntheticSample.model_validate_json(line))
    return samples


__all__ = [
    "build_dataset",
    "build_sample",
    "load_dataset",
    "save_dataset",
]
