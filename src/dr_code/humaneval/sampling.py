from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol, Self, cast

from datasets import load_dataset  # type: ignore[import-not-found]
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from dr_code.humaneval.task import (
    HUMAN_EVAL_OVERRIDES,
    HumanEvalOverride,
    HumanEvalTask,
    parse_human_eval_dataset,
)
from dr_code.eval.tasks import (
    RepeatId,
    RepeatPlan,
    SampleIdentity,
    TaskSet,
    humaneval_source_content_hash,
    humaneval_task_identity,
    sample_identity_for,
)
from dr_code.eval.lifecycle import SamplingConfig, SamplingDefinition
from dr_code.models import FrozenModel

HumanEvalRow = Mapping[str, Any]
HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION = 2
DEFAULT_HUMAN_EVAL_DATASET_NAME = "evalplus/humanevalplus"
DEFAULT_HUMAN_EVAL_DATASET_SPLIT = "test"
DEFAULT_HUMAN_EVAL_HF_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256: Final = (
    "efb5d325d225243c48fb9848feeaa1e263dc7c335b6a6030b409d3e7cbb7b422"
)


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


class SampledHumanEvalTask(FrozenModel):
    sample_index: StrictInt
    sampling_config_identity: StrictStr
    sampling_config: SamplingConfig
    task: HumanEvalTask
    repeat_id: RepeatId
    sample_identity: SampleIdentity

    @field_validator("sampling_config_identity")
    @classmethod
    def validate_sampling_identity(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(
                "sampling config identity must be a lowercase SHA-256"
            )
        return value

    @field_validator("sample_index")
    @classmethod
    def validate_sample_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sample index must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_task_identity(self) -> Self:
        task_identity = humaneval_task_identity(self.task)
        if self.repeat_id.task_identity != task_identity:
            raise ValueError(
                "embedded HumanEval task identity must match "
                "repeat_id.task_identity"
            )
        if (
            self.sampling_config_identity
            != self.sampling_config.config_identity_hash
        ):
            raise ValueError(
                "sampling config identity must match the embedded "
                "SamplingConfig"
            )
        repeats = self.sampling_config.repeat_plan.repeats()
        if self.sample_index >= len(repeats):
            raise ValueError("sample index is outside the sampling plan")
        if repeats[self.sample_index].repeat_id != self.repeat_id:
            raise ValueError(
                "sample repeat must match its ordinal in the sampling plan"
            )
        expected = sample_identity_for(
            sampling_config_identity=self.sampling_config_identity,
            repeat_id=self.repeat_id,
            ordinal=self.sample_index,
            task_identity=task_identity,
        )
        if self.sample_identity != expected:
            raise ValueError(
                "sample identity must authenticate config, repeat, ordinal, "
                "and task"
            )
        return self


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
    dataset_split: StrictStr
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
    expected_snapshot_sha256: str | None = None,
) -> list[HumanEvalRow]:
    if snapshot_path is not None:
        return load_human_eval_snapshot_rows(
            snapshot_path=Path(snapshot_path),
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            hf_revision=hf_revision,
            expected_snapshot_sha256=expected_snapshot_sha256,
        )
    if expected_snapshot_sha256 is not None:
        raise ValueError(
            "expected_snapshot_sha256 is only valid with snapshot_path"
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
    dataset_split: str = DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
    expected_snapshot_sha256: str | None = None,
) -> list[HumanEvalRow]:
    expected_digest = _trusted_snapshot_digest(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        hf_revision=hf_revision,
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    snapshot_bytes = snapshot_path.read_bytes()
    actual_digest = hashlib.sha256(snapshot_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(
            "HumanEval raw-row snapshot content digest mismatch: "
            f"{actual_digest!r} != {expected_digest!r}"
        )
    snapshot = HumanEvalRawRowsSnapshot.model_validate_json(snapshot_bytes)
    validate_snapshot_header(
        snapshot.header,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        hf_revision=hf_revision,
    )
    rows = [row.model_dump(mode="json") for row in snapshot.rows]
    parse_human_eval_dataset(rows)
    return rows


def _trusted_snapshot_digest(
    *,
    dataset_name: str,
    dataset_split: str,
    hf_revision: str,
    expected_snapshot_sha256: str | None,
) -> str:
    default_coordinates = (
        dataset_name == DEFAULT_HUMAN_EVAL_DATASET_NAME
        and dataset_split == DEFAULT_HUMAN_EVAL_DATASET_SPLIT
        and hf_revision == DEFAULT_HUMAN_EVAL_HF_REVISION
    )
    if default_coordinates:
        if (
            expected_snapshot_sha256 is not None
            and expected_snapshot_sha256 != DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256
        ):
            raise ValueError(
                "default HumanEval coordinates require the trusted "
                "canonical snapshot digest"
            )
        return DEFAULT_HUMAN_EVAL_SNAPSHOT_SHA256
    if expected_snapshot_sha256 is None:
        raise ValueError(
            "custom HumanEval snapshot coordinates require an explicit "
            "expected_snapshot_sha256"
        )
    if len(expected_snapshot_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_snapshot_sha256
    ):
        raise ValueError(
            "expected_snapshot_sha256 must be a lowercase SHA-256"
        )
    return expected_snapshot_sha256


def validate_snapshot_header(
    header: HumanEvalRawRowsSnapshotHeader,
    *,
    dataset_name: str,
    dataset_split: str,
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
    if header.dataset_split != dataset_split:
        raise ValueError(
            "HumanEval raw-row snapshot dataset split mismatch: "
            f"{header.dataset_split!r} != {dataset_split!r}"
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
    dataset_split: str = DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
) -> Path:
    snapshot = HumanEvalRawRowsSnapshot(
        header=HumanEvalRawRowsSnapshotHeader(
            schema_version=HUMAN_EVAL_RAW_ROW_SNAPSHOT_SCHEMA_VERSION,
            dataset_id=dataset_name,
            dataset_split=dataset_split,
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
    dataset_name: str,
    dataset_split: str,
    hf_revision: str,
) -> list[SampledHumanEvalTask]:
    if type(sample_count) is not int:
        raise TypeError("sample_count must be an integer")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    tasks = tuple(parse_human_eval_dataset(rows))
    if sample_count > len(tasks):
        raise ValueError(
            "sample_count exceeds the available HumanEval population: "
            f"{sample_count} > {len(tasks)}"
        )
    if sample_count == 0:
        return []
    indices = list(range(len(tasks)))
    random.Random(seed).shuffle(indices)
    selected = tuple(
        tasks[task_index] for task_index in indices[:sample_count]
    )
    task_identities = tuple(humaneval_task_identity(task) for task in selected)
    source_task_identities = tuple(
        humaneval_task_identity(task) for task in tasks
    )
    task_set = TaskSet(
        manifest_id="humaneval.seeded_sample",
        version="1",
        dataset_id=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=hf_revision,
        source_content_hash=humaneval_source_content_hash(tasks),
        source_task_identities=source_task_identities,
        task_identities=task_identities,
    )
    repeat_plan = RepeatPlan(
        plan_id="humaneval.seeded_sample",
        version="1",
        task_identities=task_identities,
        repeat_count=1,
        seeds=tuple(
            (f"{task_identity}#0", seed) for task_identity in task_identities
        ),
    )
    sampling = SamplingDefinition(
        definition_id="humaneval.seeded_sample",
        version="1",
    ).materialize(task_set=task_set, repeat_plan=repeat_plan)
    return run_human_eval_sampling(
        tasks,
        sampling=sampling,
    )


def run_human_eval_sampling(
    tasks: Sequence[HumanEvalTask],
    *,
    sampling: SamplingConfig,
) -> list[SampledHumanEvalTask]:
    """Run one concrete TaskSet and RepeatPlan over HumanEval tasks."""

    sampling = SamplingConfig.model_validate(
        sampling.model_dump(mode="python")
    )
    task_set = sampling.task_set
    repeat_plan = sampling.repeat_plan
    if repeat_plan.task_identities != task_set.task_identities:
        raise ValueError(
            "repeat plan task identities must match the TaskSet manifest"
        )
    task_by_identity: dict[str, HumanEvalTask] = {}
    for task in tasks:
        identity = humaneval_task_identity(task)
        if identity in task_by_identity:
            raise ValueError(
                f"duplicate HumanEval task identity in input: {identity}"
            )
        task_by_identity[identity] = task
    actual_source_identities = tuple(task_by_identity)
    if actual_source_identities != task_set.source_task_identities:
        raise ValueError(
            "HumanEval source population does not match TaskSet source "
            "identities"
        )
    actual_source_content_hash = humaneval_source_content_hash(
        tuple(task_by_identity.values())
    )
    if actual_source_content_hash != task_set.source_content_hash:
        raise ValueError(
            "HumanEval source population content does not match TaskSet"
        )
    missing = set(task_set.task_identities) - set(task_by_identity)
    if missing:
        raise ValueError(
            "TaskSet references missing HumanEval tasks: "
            + ", ".join(sorted(missing))
        )
    return [
        SampledHumanEvalTask(
            sample_index=sample_index,
            sampling_config_identity=sampling.config_identity_hash,
            sampling_config=sampling,
            task=task_by_identity[repeat.repeat_id.task_identity],
            repeat_id=repeat.repeat_id,
            sample_identity=sample_identity_for(
                sampling_config_identity=sampling.config_identity_hash,
                repeat_id=repeat.repeat_id,
                ordinal=sample_index,
                task_identity=repeat.repeat_id.task_identity,
            ),
        )
        for sample_index, repeat in enumerate(repeat_plan.repeats())
    ]


def sample_human_eval_tasks(
    *,
    seed: int,
    sample_count: int,
    dataset_name: str = DEFAULT_HUMAN_EVAL_DATASET_NAME,
    dataset_split: str = DEFAULT_HUMAN_EVAL_DATASET_SPLIT,
    hf_revision: str = DEFAULT_HUMAN_EVAL_HF_REVISION,
    snapshot_path: str | Path | None = None,
    expected_snapshot_sha256: str | None = None,
) -> list[SampledHumanEvalTask]:
    return sample_human_eval_tasks_from_rows(
        load_human_eval_rows(
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            hf_revision=hf_revision,
            snapshot_path=snapshot_path,
            expected_snapshot_sha256=expected_snapshot_sha256,
        ),
        seed=seed,
        sample_count=sample_count,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        hf_revision=hf_revision,
    )
