"""Build the synthetic test corpus.

Combines HumanEvalPlus canonical solutions with the frozen recipe set to
produce ~(n_tasks * n_recipes) `SyntheticSample` rows. Deterministic
given the dataset version: seeds are derived from
`(humaneval_task_id, recipe_name, dataset_version)`.

The resulting JSONL artifact is checked into `tests/code_eval/corpus/`.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

from code_eval.models.synthetic_sample import SyntheticSample
from code_eval.names import DATASET_VERSION
from code_eval.synthetic.corruption_recipes import (
    RECIPES,
    Recipe,
    apply_recipe,
)
from code_eval.synthetic.equivalence import canonicalize
from code_eval.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    load_humaneval_plus,
)


def _seed_for(task_id: str, recipe_name: str, dataset_version: str) -> int:
    """Derive a stable per-sample seed.

    Uses BLAKE2b on the concatenation of (task_id, recipe_name,
    dataset_version) and reduces the digest to a 64-bit int. The exact
    seed depends only on these three inputs.
    """
    blob = f"{task_id}|{recipe_name}|{dataset_version}".encode()
    digest = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def build_sample(
    task: HumanEvalPlusTask,
    recipe: Recipe,
    dataset_version: str = DATASET_VERSION,
) -> SyntheticSample:
    """Build a single synthetic sample for one (task, recipe) pair."""
    ground_truth = canonicalize(task.full_source)
    rng = random.Random(_seed_for(task.task_id, recipe.name, dataset_version))
    corrupted = apply_recipe(recipe, ground_truth, rng)
    return SyntheticSample(
        sample_id=SyntheticSample.make_id(task.task_id, recipe.name),
        humaneval_task_id=task.task_id,
        recipe_name=recipe.name,
        ground_truth_source=ground_truth,
        corrupted_source=corrupted.corrupted_source,
        expected_recovery_steps=corrupted.expected_recovery_steps,
        # Phase 1 doesn't yet enforce a strict ordered extractor path —
        # we record an empty tuple here, leaving it as future work.
        expected_extractor_path_contains=(),
        dataset_version=dataset_version,
    )


def iter_dataset(
    tasks: Iterable[HumanEvalPlusTask],
    recipes: Iterable[Recipe] = RECIPES,
    dataset_version: str = DATASET_VERSION,
) -> Iterator[SyntheticSample]:
    """Yield one `SyntheticSample` per (task, recipe) pair."""
    recipes_list = list(recipes)
    for task in tasks:
        for recipe in recipes_list:
            yield build_sample(task, recipe, dataset_version)


def build_dataset(
    tasks: Iterable[HumanEvalPlusTask] | None = None,
    recipes: Iterable[Recipe] = RECIPES,
    dataset_version: str = DATASET_VERSION,
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
    return list(iter_dataset(tasks_iter, recipes, dataset_version))


def save_dataset(samples: Iterable[SyntheticSample], path: Path) -> int:
    """Serialize samples as JSONL to `path`. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            # frozenset isn't JSON-serializable directly; convert to sorted list.
            payload = sample.model_dump()
            payload["expected_recovery_steps"] = sorted(payload["expected_recovery_steps"])
            payload["expected_extractor_path_contains"] = list(
                payload["expected_extractor_path_contains"]
            )
            fh.write(json.dumps(payload, ensure_ascii=False))
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
            payload = json.loads(line)
            payload["expected_recovery_steps"] = frozenset(
                payload.get("expected_recovery_steps", [])
            )
            payload["expected_extractor_path_contains"] = tuple(
                payload.get("expected_extractor_path_contains", [])
            )
            samples.append(SyntheticSample(**payload))
    return samples


DEFAULT_DATASET_PATH: Final[Path] = (
    Path("tests") / "code_eval" / "corpus" / "synthetic_dataset.jsonl"
)


__all__ = [
    "DEFAULT_DATASET_PATH",
    "build_dataset",
    "build_sample",
    "iter_dataset",
    "load_dataset",
    "save_dataset",
]
