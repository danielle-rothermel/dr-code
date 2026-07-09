"""Load HumanEvalPlus ground-truth solutions from Hugging Face.

The "Plus" variant matters because it ships extended unit tests — useful
for future opt-in execution-based equivalence checks. The plain
`canonical_solution` + `prompt` text is used for our syntactic ground truth.

If network access is unavailable, callers must explicitly opt into the offline
JSON snapshot under `tests/corpus/humanevalplus_snapshot.json`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from dr_code.synthetic.models import FrozenModel

#: Hugging Face dataset id and split.
HF_DATASET_ID: Final[str] = "evalplus/humanevalplus"
HF_SPLIT: Final[str] = "test"

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


TASK_LIST_ADAPTER: Final[TypeAdapter[list[HumanEvalPlusTask]]] = TypeAdapter(
    list[HumanEvalPlusTask]
)


def _try_load_from_hf() -> list[HumanEvalPlusTask] | None:
    """Attempt to load from Hugging Face. Returns None on any failure."""
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
                prompt=row["prompt"],
                canonical_solution=row["canonical_solution"],
                entry_point=row["entry_point"],
                test=row.get("test", ""),
            )
        )
    return tasks


def _try_load_from_snapshot(repo_root: Path) -> list[HumanEvalPlusTask] | None:
    """Attempt to load from the local snapshot file. Returns None if missing."""
    snap = repo_root / SNAPSHOT_REL_PATH
    if not snap.exists():
        return None
    return TASK_LIST_ADAPTER.validate_json(snap.read_text(encoding="utf-8"))


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
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(
        TASK_LIST_ADAPTER.dump_json(tasks, indent=2).decode(),
        encoding="utf-8",
    )
    return snap


def load_humaneval_plus(
    prefer_snapshot: bool = False,
) -> list[HumanEvalPlusTask]:
    """Load HumanEvalPlus tasks.

    Args:
        prefer_snapshot: If True, try the local snapshot first. Default is
            to require a Hugging Face load, so stale snapshots are never used
            silently when the network path fails.

    Raises:
        FileNotFoundError: If the selected source is unavailable.
    """
    repo_root = _repo_root()
    if prefer_snapshot:
        tasks = _try_load_from_snapshot(repo_root) or _try_load_from_hf()
    else:
        tasks = _try_load_from_hf()
    if tasks is None:
        raise FileNotFoundError(
            "HumanEvalPlus unavailable from the selected source. "
            "Pass prefer_snapshot=True to use the checked-in snapshot at "
            f"{SNAPSHOT_REL_PATH}."
        )
    return tasks
