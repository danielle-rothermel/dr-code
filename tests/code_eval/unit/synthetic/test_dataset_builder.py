"""Tests for the synthetic dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

from code_eval.names import DATASET_VERSION
from code_eval.synthetic import (
    DEFAULT_DATASET_PATH,
    RECIPES,
    RECIPES_BY_NAME,
    build_dataset,
    build_sample,
    load_dataset,
    load_humaneval_plus,
    save_dataset,
)


def test_recipe_set_is_unique() -> None:
    names = [r.name for r in RECIPES]
    assert len(names) == len(set(names)), f"duplicate recipe names: {names}"
    assert set(names) == set(RECIPES_BY_NAME.keys())


def test_dataset_size_matches_cross_product() -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)
    samples = build_dataset(tasks=tasks)
    assert len(samples) == len(tasks) * len(RECIPES)


def test_dataset_is_deterministic() -> None:
    """Two builds with identical inputs produce identical samples."""
    tasks = load_humaneval_plus(prefer_snapshot=True)
    a = build_dataset(tasks=tasks)
    b = build_dataset(tasks=tasks)
    for x, y in zip(a, b, strict=True):
        assert x == y


def test_dataset_jsonl_roundtrip(tmp_path: Path) -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)[:2]
    samples = build_dataset(tasks=tasks)
    out = tmp_path / "ds.jsonl"
    n = save_dataset(samples, out)
    assert n == len(samples)
    back = load_dataset(out)
    for x, y in zip(samples, back, strict=True):
        assert x == y


def test_default_dataset_artifact_exists() -> None:
    """Phase 1 commits a built JSONL under tests/corpus/."""
    # Walk up from this file to find the repo root.
    here = Path(__file__).resolve()
    repo_root = next(p for p in here.parents if (p / "pyproject.toml").exists())
    artifact = repo_root / DEFAULT_DATASET_PATH
    assert artifact.exists(), f"missing {artifact}"
    # spot-check: first row has the expected shape and current dataset version.
    first_line = artifact.open("r", encoding="utf-8").readline()
    payload = json.loads(first_line)
    assert payload["dataset_version"] == DATASET_VERSION
    assert "::" in payload["sample_id"]


def test_build_sample_seeding_is_stable() -> None:
    """The same (task, recipe) pair always produces the same corrupted source."""
    tasks = load_humaneval_plus(prefer_snapshot=True)
    task = tasks[0]
    for recipe in RECIPES:
        a = build_sample(task, recipe)
        b = build_sample(task, recipe)
        assert a == b
