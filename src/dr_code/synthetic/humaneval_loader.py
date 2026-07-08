"""Load HumanEvalPlus ground-truth solutions.

The "Plus" variant matters because it ships extended unit tests — useful
for future opt-in execution-based equivalence checks. The plain
`canonical_solution` + `prompt` text is used for our syntactic ground truth.

If network access is unavailable, callers must explicitly opt into the offline
JSON snapshot under `tests/corpus/humanevalplus_snapshot.json`.
The raw-row loading contract is owned by `dr_code.humaneval.sampling`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dr_code.humaneval.sampling import (
    DEFAULT_HUMAN_EVAL_DATASET_NAME,
    DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    DEFAULT_HUMAN_EVAL_HF_REVISION,
    HumanEvalRow,
    load_human_eval_rows,
    write_human_eval_snapshot_rows,
)
from dr_code.synthetic.models import FrozenModel

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


def _load_from_snapshot(repo_root: Path) -> list[HumanEvalPlusTask]:
    snap = repo_root / SNAPSHOT_REL_PATH
    rows = load_human_eval_rows(
        dataset_name=HF_DATASET_ID,
        dataset_split=HF_SPLIT,
        hf_revision=HF_REVISION,
        snapshot_path=snap,
    )
    return [_task_from_row(row) for row in rows]


def _repo_root() -> Path:
    """Walk up from this file to the repo root (where pyproject.toml lives)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def save_snapshot(
    tasks: list[HumanEvalPlusTask], repo_root: Path | None = None
) -> Path:
    """Write a snapshot to disk for offline reuse. Returns the path written."""
    root = repo_root or _repo_root()
    snap = root / SNAPSHOT_REL_PATH
    return write_human_eval_snapshot_rows(
        [task.model_dump(mode="json") for task in tasks],
        snapshot_path=snap,
        dataset_name=HF_DATASET_ID,
        hf_revision=HF_REVISION,
    )


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
