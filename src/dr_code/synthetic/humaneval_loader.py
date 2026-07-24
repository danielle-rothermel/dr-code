"""Load HumanEvalPlus ground-truth solutions.

The packaged snapshot is the offline default. Callers may explicitly select
the independently loaded pinned Hugging Face revision.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Final

from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_DATASET_NAME,
    DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    DEFAULT_HUMAN_EVAL_HF_REVISION,
    HumanEvalRow,
    load_human_eval_rows,
    load_human_eval_snapshot_rows_bytes,
    write_human_eval_snapshot_rows,
)
from dr_code.models import FrozenModel

#: Hugging Face dataset id and split.
HF_DATASET_ID: Final[str] = DEFAULT_HUMAN_EVAL_DATASET_NAME
HF_SPLIT: Final[str] = DEFAULT_HUMAN_EVAL_DATASET_SPLIT
HF_REVISION: Final[str] = DEFAULT_HUMAN_EVAL_HF_REVISION

SNAPSHOT_RESOURCE: Final[str] = "humanevalplus_snapshot.json"
SNAPSHOT_SHA256: Final[str] = (
    "efb5d325d225243c48fb9848feeaa1e263dc7c335b6a6030b409d3e7cbb7b422"
)


class HumanEvalSource(StrEnum):
    """The two independent pinned HumanEval+ sources."""

    SNAPSHOT = "snapshot"
    HF = "hf"


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


def _task_from_row(row: HumanEvalRow) -> HumanEvalPlusTask:
    return HumanEvalPlusTask(
        task_id=str(row["task_id"]),
        prompt=str(row["prompt"]),
        canonical_solution=str(row["canonical_solution"]),
        entry_point=str(row["entry_point"]),
        test=str(row["test"]),
    )


def _load_from_hf() -> list[HumanEvalPlusTask]:
    rows = load_human_eval_rows(
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
    )
    return [_task_from_row(row) for row in rows]


def packaged_snapshot_bytes() -> bytes:
    """Read the canonical snapshot from the installed package."""

    content = (
        files("dr_code.synthetic").joinpath(SNAPSHOT_RESOURCE).read_bytes()
    )
    if hashlib.sha256(content).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("packaged HumanEval+ snapshot SHA-256 mismatch")
    return content


def _load_from_snapshot() -> list[HumanEvalPlusTask]:
    rows = load_human_eval_snapshot_rows_bytes(
        packaged_snapshot_bytes(),
        dataset_name=HF_DATASET_ID,
        hf_revision=HF_REVISION,
    )
    return [_task_from_row(row) for row in rows]


def save_snapshot(
    tasks: list[HumanEvalPlusTask],
    destination: Path,
) -> Path:
    """Write snapshot rows to an explicit tooling destination."""

    return write_human_eval_snapshot_rows(
        [task.model_dump(mode="json") for task in tasks],
        snapshot_path=destination,
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
    )


def load_humaneval_plus(
    source: HumanEvalSource = HumanEvalSource.SNAPSHOT,
) -> list[HumanEvalPlusTask]:
    """Load the packaged snapshot or an explicitly selected pinned HF source."""

    if source is HumanEvalSource.SNAPSHOT:
        return _load_from_snapshot()
    if source is HumanEvalSource.HF:
        return _load_from_hf()
    raise ValueError(f"unsupported HumanEval+ source: {source!r}")
