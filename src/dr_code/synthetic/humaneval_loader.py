"""Load HumanEvalPlus ground-truth solutions.

The "Plus" variant ships extended unit tests, which are carried on the task
model's `test` field. The plain `canonical_solution` + `prompt` text is what
is used for our syntactic ground truth.

If network access is unavailable, callers must explicitly opt into an offline
JSON snapshot by passing its path; this module invents no location of its own.
The raw-row loading contract is owned by `dr_code.humaneval.sampling`, which
guarantees provenance only.

This loader validates rows against the registered task model
(`parse_humaneval_dataset`) before building synthetic tasks, so a row that
fails the registered override set — for example an overridden task whose
replacement anchor is missing — never becomes synthetic ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from dr_code.humaneval.sampling import (
    DEFAULT_HUMANEVAL_DATASET_NAME,
    DEFAULT_HUMANEVAL_DATASET_SPLIT,
    DEFAULT_HUMANEVAL_HF_REVISION,
    HumanEvalRow,
    load_humaneval_rows,
)
from dr_code.humaneval.task import HumanEvalTask, parse_humaneval_dataset
from dr_code.base import FrozenModel

#: Hugging Face dataset id and split.
HF_DATASET_ID: Final[str] = DEFAULT_HUMANEVAL_DATASET_NAME
HF_SPLIT: Final[str] = DEFAULT_HUMANEVAL_DATASET_SPLIT
HF_REVISION: Final[str] = DEFAULT_HUMANEVAL_HF_REVISION


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

    ``parse_humaneval_dataset`` applies the registered override set, so a
    row that lost its override anchor raises here instead of silently
    becoming synthetic ground truth.
    """
    return [
        _task_from_validated(task) for task in parse_humaneval_dataset(rows)
    ]


def _load_from_hf() -> list[HumanEvalPlusTask]:
    rows = load_humaneval_rows(
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
    )
    return _tasks_from_rows(rows)


def _load_from_snapshot(snapshot_path: Path) -> list[HumanEvalPlusTask]:
    rows = load_humaneval_rows(
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
        snapshot_path=snapshot_path,
    )
    return _tasks_from_rows(rows)


def load_humaneval_plus(
    prefer_snapshot: bool = False,
    snapshot_path: Path | None = None,
) -> list[HumanEvalPlusTask]:
    """Load HumanEvalPlus tasks.

    Args:
        prefer_snapshot: If True, load ``snapshot_path``. Default loads the
            pinned Hugging Face revision, so stale snapshots are never used
            silently when the network path fails.
        snapshot_path: The offline snapshot to read. Required when
            ``prefer_snapshot`` is True; this module knows no default
            location.

    Raises:
        ValueError: If ``prefer_snapshot`` is True without a snapshot path.
        FileNotFoundError: If the selected source is unavailable.
    """
    if prefer_snapshot:
        if snapshot_path is None:
            raise ValueError(
                "prefer_snapshot=True requires an explicit snapshot_path"
            )
        return _load_from_snapshot(snapshot_path)
    return _load_from_hf()
