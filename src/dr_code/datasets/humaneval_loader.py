"""Load HumanEvalPlus tasks from Hugging Face or offline snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from dr_code.models.humaneval import (
    HumanEvalPlusTask,
    expected_arity_from_prompt,
)

HF_DATASET_ID: Final[str] = "evalplus/humanevalplus"
HF_SPLIT: Final[str] = "test"
SNAPSHOT_REL_PATH: Final[str] = "tests/corpus/humanevalplus_snapshot.json"


def _try_load_from_hf() -> list[HumanEvalPlusTask] | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    except Exception:
        return None
    tasks: list[HumanEvalPlusTask] = []
    for row in ds:
        tasks.append(
            HumanEvalPlusTask(
                task_id=row["task_id"],
                entry_point=row["entry_point"],
                prompt=row["prompt"],
                canonical_solution=row["canonical_solution"],
                test=row["test"],
                expected_arity=expected_arity_from_prompt(row["prompt"]),
            )
        )
    return tasks


def _try_load_from_snapshot(repo_root: Path) -> list[HumanEvalPlusTask] | None:
    snap = repo_root / SNAPSHOT_REL_PATH
    if not snap.exists():
        return None
    raw = json.loads(snap.read_text(encoding="utf-8"))
    return [HumanEvalPlusTask.model_validate(record) for record in raw]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def save_snapshot(
    tasks: list[HumanEvalPlusTask],
    repo_root: Path | None = None,
) -> Path:
    """Write a snapshot to disk for offline reuse."""
    root = repo_root or _repo_root()
    snap = root / SNAPSHOT_REL_PATH
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(
        json.dumps(
            [task.model_dump() for task in tasks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return snap


def load_humaneval_plus(
    prefer_snapshot: bool = True,
) -> list[HumanEvalPlusTask]:
    """Load HumanEvalPlus tasks from snapshot or Hugging Face."""
    repo_root = _repo_root()
    if prefer_snapshot:
        tasks = _try_load_from_snapshot(repo_root) or _try_load_from_hf()
    else:
        tasks = _try_load_from_hf() or _try_load_from_snapshot(repo_root)
    if tasks is None:
        msg = (
            "HumanEvalPlus unavailable: no Hugging Face access and no snapshot at "
            f"{SNAPSHOT_REL_PATH}. Run "
            "`uv run scripts/build_humaneval_snapshot.py` from a machine with "
            "network access to create one."
        )
        raise FileNotFoundError(msg)
    return tasks


def task_index(
    tasks: list[HumanEvalPlusTask] | None = None,
    *,
    prefer_snapshot: bool = True,
) -> dict[str, HumanEvalPlusTask]:
    """Return task_id → task mapping."""
    loaded = (
        tasks if tasks is not None else load_humaneval_plus(prefer_snapshot)
    )
    return {task.task_id: task for task in loaded}


def get_task(
    task_id: str,
    *,
    prefer_snapshot: bool = True,
) -> HumanEvalPlusTask:
    """Return a single task by id."""
    index = task_index(prefer_snapshot=prefer_snapshot)
    if task_id not in index:
        msg = f"Unknown HumanEval+ task_id: {task_id}"
        raise KeyError(msg)
    return index[task_id]
