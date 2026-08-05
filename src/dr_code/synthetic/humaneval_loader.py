"""Load HumanEvalPlus ground-truth solutions.

The "Plus" variant ships extended unit tests, which are carried on the task
model's `test` field. The plain `canonical_solution` + `prompt` text is what
is used for our syntactic ground truth.

If network access is unavailable, callers must explicitly opt into the offline
JSON snapshot under `tests/corpus/humanevalplus_snapshot.json`.
The raw-row loading contract is owned by `dr_code.humaneval.sampling`, which
guarantees provenance only.

This loader validates rows against the registered task model
(`parse_human_eval_dataset`) before building synthetic tasks, so a row that
fails the registered override set — for example an overridden task whose
replacement anchor is missing — never becomes synthetic ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_DATASET_NAME,
    DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    DEFAULT_HUMAN_EVAL_HF_REVISION,
    HumanEvalRow,
    load_human_eval_rows,
)
from dr_code.humaneval.task import HumanEvalTask, parse_human_eval_dataset
from dr_code.models import FrozenModel

#: Hugging Face dataset id and split.
HF_DATASET_ID: Final[str] = DEFAULT_HUMAN_EVAL_DATASET_NAME
HF_SPLIT: Final[str] = DEFAULT_HUMAN_EVAL_DATASET_SPLIT
HF_REVISION: Final[str] = DEFAULT_HUMAN_EVAL_HF_REVISION

#: Path to the offline snapshot, relative to repo root.
SNAPSHOT_REL_PATH: Final[str] = "tests/corpus/humanevalplus_snapshot.json"


class HumanEvalPlusTask(FrozenModel):
    """One task from HumanEvalPlus."""

    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str

    @property
    def full_source(self) -> str:
        """Return the full ground-truth program (prompt + solution body)."""
        return self.prompt + self.canonical_solution


def _task_from_validated(task: HumanEvalTask) -> HumanEvalPlusTask:
    return HumanEvalPlusTask(
        task_id=task.task_id,
        prompt=task.prompt,
        canonical_solution=task.canonical_solution,
        entry_point=task.entry_point,
        test=task.test,
    )


def _tasks_from_rows(rows: Sequence[HumanEvalRow]) -> list[HumanEvalPlusTask]:
    """Validate rows against the registered task model, then project them.

    ``parse_human_eval_dataset`` applies the registered override set, so a
    row that lost its override anchor raises here instead of silently
    becoming synthetic ground truth.
    """
    return [
        _task_from_validated(task) for task in parse_human_eval_dataset(rows)
    ]


def _load_from_hf() -> list[HumanEvalPlusTask]:
    rows = load_human_eval_rows(
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
    )
    return _tasks_from_rows(rows)


def _load_from_snapshot(repo_root: Path) -> list[HumanEvalPlusTask]:
    snap = repo_root / SNAPSHOT_REL_PATH
    rows = load_human_eval_rows(
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
        snapshot_path=snap,
    )
    return _tasks_from_rows(rows)


def _repo_root() -> Path:
    """Walk up from this file to the repo root (where pyproject.toml lives)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_humaneval_plus(
    prefer_snapshot: bool = False,
) -> list[HumanEvalPlusTask]:
    """Load HumanEvalPlus tasks.

    Args:
        prefer_snapshot: If True, load the local snapshot. Default loads the
            pinned Hugging Face revision, so stale snapshots are never used
            silently when the network path fails.

    Raises:
        FileNotFoundError: If the selected source is unavailable.
    """
    repo_root = _repo_root()
    if prefer_snapshot:
        return _load_from_snapshot(repo_root)
    return _load_from_hf()
