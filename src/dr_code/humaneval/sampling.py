from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from datasets import load_dataset  # type: ignore[import-not-found]
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

from dr_code.humaneval.task import (
    HUMAN_EVAL_OVERRIDES,
    HumanEvalOverride,
    HumanEvalTask,
    parse_human_eval_dataset,
)

HumanEvalRow = Mapping[str, Any]
HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION = 1
DEFAULT_HUMAN_EVAL_DATASET_NAME = "evalplus/humanevalplus"
DEFAULT_HUMAN_EVAL_DATASET_SPLIT = "test"
DEFAULT_HUMAN_EVAL_HF_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


class SampledHumanEvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_index: StrictInt
    task: HumanEvalTask


class HumanEvalRawRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: StrictStr
    prompt: StrictStr
    canonical_solution: StrictStr
    entry_point: StrictStr
    test: StrictStr


class HumanEvalRawRowsSnapshotHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: StrictInt
    dataset_id: StrictStr
    hf_revision: StrictStr
    overrides_digest: StrictStr


class HumanEvalRawRowsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header: HumanEvalRawRowsSnapshotHeader
    rows: list[HumanEvalRawRow]


def human_eval_overrides_digest(
    overrides: dict[str, HumanEvalOverride] | None = None,
) -> str:
    active_overrides = HUMAN_EVAL_OVERRIDES if overrides is None else overrides
    payload = {
        task_id: override.model_dump(mode="json")
        for task_id, override in sorted(active_overrides.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def load_human_eval_rows(
    *,
    dataset_name: str = DEFAULT_HUMAN_EVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
    snapshot_path: str | Path | None = None,
) -> list[HumanEvalRow]:
    if snapshot_path is not None:
        return load_human_eval_snapshot_rows(
            snapshot_path=Path(snapshot_path),
            dataset_name=dataset_name,
            hf_revision=hf_revision,
        )

    dataset = cast(
        HumanEvalDataset,
        load_dataset(dataset_name, split=dataset_split, revision=hf_revision),
    )
    return [dataset[index] for index in range(len(dataset))]


def load_human_eval_snapshot_rows(
    *,
    snapshot_path: Path,
    dataset_name: str = DEFAULT_HUMAN_EVAL_DATASET_NAME,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
) -> list[HumanEvalRow]:
    snapshot = HumanEvalRawRowsSnapshot.model_validate_json(
        snapshot_path.read_text(encoding="utf-8")
    )
    validate_snapshot_header(
        snapshot.header,
        dataset_name=dataset_name,
        hf_revision=hf_revision,
    )
    rows = [row.model_dump(mode="json") for row in snapshot.rows]
    parse_human_eval_dataset(rows)
    return rows


def validate_snapshot_header(
    header: HumanEvalRawRowsSnapshotHeader,
    *,
    dataset_name: str,
    hf_revision: str,
) -> None:
    if header.schema_version != HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported HumanEval raw-row snapshot schema version: "
            f"{header.schema_version}"
        )
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
    expected_overrides_digest = human_eval_overrides_digest()
    if header.overrides_digest != expected_overrides_digest:
        raise ValueError(
            "HumanEval raw-row snapshot overrides digest mismatch: "
            f"{header.overrides_digest!r} != {expected_overrides_digest!r}"
        )


def write_human_eval_snapshot_rows(
    rows: Sequence[HumanEvalRow],
    *,
    snapshot_path: Path,
    dataset_name: str = DEFAULT_HUMAN_EVAL_DATASET_NAME,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
) -> Path:
    snapshot = HumanEvalRawRowsSnapshot(
        header=HumanEvalRawRowsSnapshotHeader(
            schema_version=HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION,
            dataset_id=dataset_name,
            hf_revision=hf_revision,
            overrides_digest=human_eval_overrides_digest(),
        ),
        rows=[HumanEvalRawRow.model_validate(row) for row in rows],
    )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        snapshot.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return snapshot_path


def sample_human_eval_tasks_from_rows(
    rows: Sequence[HumanEvalRow],
    *,
    seed: int,
    sample_count: int,
) -> list[SampledHumanEvalTask]:
    tasks = parse_human_eval_dataset(rows)
    indices = list(range(len(tasks)))
    random.Random(seed).shuffle(indices)
    return [
        SampledHumanEvalTask(sample_index=sample_index, task=tasks[task_index])
        for sample_index, task_index in enumerate(indices[:sample_count])
    ]


def sample_human_eval_tasks(
    *,
    seed: int,
    sample_count: int,
    dataset_name: str = DEFAULT_HUMAN_EVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
    snapshot_path: str | Path | None = None,
) -> list[SampledHumanEvalTask]:
    return sample_human_eval_tasks_from_rows(
        load_human_eval_rows(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            hf_revision=hf_revision,
            snapshot_path=snapshot_path,
        ),
        seed=seed,
        sample_count=sample_count,
    )
