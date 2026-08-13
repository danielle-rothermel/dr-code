from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from drc_humaneval.sampling import (
    DEFAULT_HUMANEVAL_DATASET_NAME,
    DEFAULT_HUMANEVAL_DATASET_SPLIT,
    DEFAULT_HUMANEVAL_HF_REVISION,
    HumanEvalRow,
    load_humaneval_rows,
)
from drc_humaneval.task import HumanEvalTask, parse_humaneval_dataset
from dr_code.core.models import FrozenModel

HF_DATASET_ID: Final[str] = DEFAULT_HUMANEVAL_DATASET_NAME
HF_SPLIT: Final[str] = DEFAULT_HUMANEVAL_DATASET_SPLIT
HF_REVISION: Final[str] = DEFAULT_HUMANEVAL_HF_REVISION


class HumanEvalPlusTask(FrozenModel):
    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str

    @property
    def full_source(self) -> str:
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
    if prefer_snapshot:
        if snapshot_path is None:
            raise ValueError(
                "prefer_snapshot=True requires an explicit snapshot_path"
            )
        return _load_from_snapshot(snapshot_path)
    return _load_from_hf()
