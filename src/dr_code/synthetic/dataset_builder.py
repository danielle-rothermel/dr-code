"""Build the synthetic test corpus."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from dr_code.synthetic.models import SyntheticSample
from dr_code.synthetic.corruption_recipes import (
    RECIPES,
    Recipe,
    apply_recipe,
)
from dr_code.synthetic.equivalence import canonicalize
from dr_code.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    load_humaneval_plus,
)


def _seed_for(task_id: str, recipe_name: str, seed: int) -> int:
    """Derive a stable per-sample seed.

    Uses BLAKE2b on the concatenation of (task_id, recipe_name, seed) and
    reduces the digest to a 64-bit int.
    """
    blob = f"{task_id}|{recipe_name}|{seed}".encode()
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def build_sample(
    task: HumanEvalPlusTask,
    recipe: Recipe,
    seed: int,
) -> SyntheticSample:
    """Build a single synthetic sample for one (task, recipe) pair."""
    ground_truth = canonicalize(task.full_source)
    rng = random.Random(_seed_for(task.task_id, recipe.name, seed))
    corrupted = apply_recipe(recipe, ground_truth, rng)
    return SyntheticSample(
        sample_id=SyntheticSample.make_id(task.task_id, recipe.name),
        humaneval_task_id=task.task_id,
        recipe_name=recipe.name,
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
) -> list[SyntheticSample]:
    """Build the full dataset list (in-memory).

    If `tasks` is None, fall back to `load_humaneval_plus()` which prefers
    the on-disk snapshot by default for offline reproducibility.
    """
    if tasks is None:
        tasks_iter: Iterable[HumanEvalPlusTask] = load_humaneval_plus(
            prefer_snapshot=prefer_snapshot
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
    "iter_dataset",
    "load_dataset",
    "save_dataset",
]
