from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from datasets import load_dataset  # type: ignore[import-not-found]
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_code.humaneval.task import (
    HumanEvalOverrideSetCoordinate,
    HumanEvalTask,
    parse_humaneval_dataset,
    resolve_humaneval_override_set,
)
from dr_code.core.models import FrozenModel

HumanEvalRow = Mapping[str, Any]
HUMANEVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION = 2
DEFAULT_HUMANEVAL_DATASET_NAME = "evalplus/humanevalplus"
DEFAULT_HUMANEVAL_DATASET_SPLIT = "test"
DEFAULT_HUMANEVAL_HF_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


class SampledHumanEvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: StrictInt
    task: HumanEvalTask


class HumanEvalRawRow(FrozenModel):
    task_id: StrictStr
    prompt: StrictStr
    canonical_solution: StrictStr
    entry_point: StrictStr
    test: StrictStr


class HumanEvalRawRowsSnapshotHeader(FrozenModel):
    schema_version: Literal[2]
    dataset_id: StrictStr
    hf_revision: StrictStr
    override_set: HumanEvalOverrideSetCoordinate


class HumanEvalRawRowsSnapshot(FrozenModel):
    header: HumanEvalRawRowsSnapshotHeader
    rows: tuple[HumanEvalRawRow, ...]


def load_humaneval_rows(
    *,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMANEVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMANEVAL_HF_REVISION,
    snapshot_path: str | Path | None = None,
) -> list[HumanEvalRow]:
    if snapshot_path is not None:
        return load_humaneval_snapshot_rows(
            snapshot_path=Path(snapshot_path),
            dataset_name=dataset_name,
            hf_revision=hf_revision,
        )

    dataset = cast(
        HumanEvalDataset,
        load_dataset(dataset_name, split=dataset_split, revision=hf_revision),
    )
    return [dataset[index] for index in range(len(dataset))]


def load_humaneval_snapshot_rows(
    *,
    snapshot_path: Path,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    hf_revision: str = DEFAULT_HUMANEVAL_HF_REVISION,
) -> list[HumanEvalRow]:
    """Return the snapshot's raw rows after checking its provenance header.

    Row-level validation belongs to whichever task model a caller builds:
    ``parse_humaneval_dataset`` for evaluation tasks, a caller's own model
    otherwise. This loader only guarantees that the rows come from the
    expected dataset, revision, and registered override set.
    """
    snapshot = HumanEvalRawRowsSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    validate_snapshot_header(
        snapshot.header,
        dataset_name=dataset_name,
        hf_revision=hf_revision,
    )
    return [row.model_dump(mode="json") for row in snapshot.rows]


def validate_snapshot_header(
    header: HumanEvalRawRowsSnapshotHeader,
    *,
    dataset_name: str,
    hf_revision: str,
) -> HumanEvalOverrideSetCoordinate:
    if header.dataset_id != dataset_name:
        raise ValueError(
            "HumanEval raw-row snapshot dataset mismatch: "
            f"{header.dataset_id!r} != {dataset_name!r}"
        )
    if header.hf_revision != hf_revision:
        raise ValueError(
            "HumanEval raw-row snapshot HF revision mismatch: "
            f"{header.hf_revision!r} != {hf_revision!r}"
        )
    registered = resolve_humaneval_override_set(
        override_set_id=header.override_set.override_set_id,
        override_set_version=header.override_set.version,
    )
    if header.override_set != registered:
        raise ValueError(
            "HumanEval raw-row snapshot override-set mismatch: "
            f"{header.override_set!r} != {registered!r}"
        )
    return registered


def sample_humaneval_tasks_from_rows(
    rows: Sequence[HumanEvalRow],
    *,
    seed: int,
    sample_count: int,
) -> list[SampledHumanEvalTask]:
    tasks = parse_humaneval_dataset(rows)
    indices = list(range(len(tasks)))
    random.Random(seed).shuffle(indices)
    return [
        SampledHumanEvalTask(sample_index=sample_index, task=tasks[task_index])
        for sample_index, task_index in enumerate(indices[:sample_count])
    ]


def sample_humaneval_tasks(
    *,
    seed: int,
    sample_count: int,
    dataset_name: str = DEFAULT_HUMANEVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMANEVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMANEVAL_HF_REVISION,
    snapshot_path: str | Path | None = None,
) -> list[SampledHumanEvalTask]:
    return sample_humaneval_tasks_from_rows(
        load_humaneval_rows(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            hf_revision=hf_revision,
            snapshot_path=snapshot_path,
        ),
        seed=seed,
        sample_count=sample_count,
    )
