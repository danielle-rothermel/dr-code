"""Run-scoped HumanEval+ snapshot cache for mutant tests."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from filelock import FileLock

from dr_code.synthetic.humaneval_loader import HumanEvalPlusTask

HumanEvalLoader = Callable[[bool], list[HumanEvalPlusTask]]

_CACHE_FILENAME = "dr-code-mutants-humaneval-plus.pickle"
_LOCK_FILENAME = f"{_CACHE_FILENAME}.lock"


def load_snapshot_tasks(
    *,
    original_loader: HumanEvalLoader,
    shared_temp_root: Path | None,
) -> tuple[HumanEvalPlusTask, ...]:
    """Load once in-process, or once across xdist workers for this pytest run."""
    if shared_temp_root is None:
        return tuple(original_loader(True))

    cache_path = shared_temp_root / _CACHE_FILENAME
    lock_path = shared_temp_root / _LOCK_FILENAME
    with FileLock(lock_path):
        if cache_path.exists():
            return _read_tasks(cache_path)

        tasks = tuple(original_loader(True))
        _write_tasks_atomically(cache_path, tasks)
        return tasks


def build_cached_loader(
    *,
    snapshot_tasks: tuple[HumanEvalPlusTask, ...],
    original_loader: HumanEvalLoader,
) -> HumanEvalLoader:
    """Return a loader preserving source choice and fresh-list semantics."""

    def cached_loader(
        prefer_snapshot: bool = False,
    ) -> list[HumanEvalPlusTask]:
        if not prefer_snapshot:
            return original_loader(False)
        return list(snapshot_tasks)

    return cached_loader


def _read_tasks(cache_path: Path) -> tuple[HumanEvalPlusTask, ...]:
    with cache_path.open("rb") as cache_file:
        value = pickle.load(cache_file)  # noqa: S301 - private run-scoped cache
    if not isinstance(value, tuple) or not all(
        isinstance(task, HumanEvalPlusTask) for task in value
    ):
        raise TypeError(f"invalid HumanEval+ task cache: {cache_path}")
    return cast(tuple[HumanEvalPlusTask, ...], value)


def _write_tasks_atomically(
    cache_path: Path,
    tasks: tuple[HumanEvalPlusTask, ...],
) -> None:
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temp_file:
            pickle.dump(tasks, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(cache_path)
    finally:
        temp_path.unlink(missing_ok=True)
