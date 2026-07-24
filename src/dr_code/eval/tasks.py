"""HumanEval task identity, Task Sets, and deliberate Repeat plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from pydantic import StrictInt, field_validator, model_validator

from dr_code.eval.identity import (
    SCHEMA_HUMANEVAL_SOURCE,
    SCHEMA_HUMANEVAL_TASK,
    SCHEMA_REPEAT_ID,
    SCHEMA_REPEAT_PLAN,
    SCHEMA_TASK_SET,
    identity_hash_for,
)
from dr_code.models import FrozenModel
from dr_code.trace.observation import SampleIdentity, sample_identity_hash

if TYPE_CHECKING:
    from dr_code.humaneval.task import HumanEvalTask

_HUMANEVAL_IDENTITY_FIELDS = (
    "task_id",
    "prompt",
    "canonical_solution",
    "entry_point",
    "test",
)


def humaneval_task_identity_payload(task: HumanEvalTask) -> dict[str, str]:
    """Return the dataset-defining fields of one HumanEval task."""

    return {
        field: getattr(task, field) for field in _HUMANEVAL_IDENTITY_FIELDS
    }


def humaneval_task_identity(task: HumanEvalTask) -> str:
    return identity_hash_for(
        schema=SCHEMA_HUMANEVAL_TASK,
        payload=humaneval_task_identity_payload(task),
    )


def humaneval_source_content_hash(tasks: tuple[HumanEvalTask, ...]) -> str:
    """Authenticate the complete ordered HumanEval source population."""

    if not tasks:
        raise ValueError("HumanEval source population must not be empty")
    return identity_hash_for(
        schema=SCHEMA_HUMANEVAL_SOURCE,
        payload=[humaneval_task_identity_payload(task) for task in tasks],
    )


class TaskSet(FrozenModel):
    """An ordered, concrete task manifest."""

    manifest_id: str
    version: str
    dataset_id: str
    dataset_split: str
    dataset_revision: str
    source_content_hash: str
    source_task_identities: tuple[str, ...]
    task_identities: tuple[str, ...]

    @field_validator("source_content_hash")
    @classmethod
    def validate_source_content_hash(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("source content hash must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if not self.source_task_identities:
            raise ValueError("source_task_identities must not be empty")
        if len(set(self.source_task_identities)) != len(
            self.source_task_identities
        ):
            raise ValueError("source_task_identities must be unique")
        if not self.task_identities:
            raise ValueError("task_identities must not be empty")
        if len(set(self.task_identities)) != len(self.task_identities):
            raise ValueError("task_identities must be unique")
        unknown = set(self.task_identities) - set(self.source_task_identities)
        if unknown:
            raise ValueError(
                "task_identities must belong to the authenticated source "
                "population"
            )
        return self

    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "dataset_id": self.dataset_id,
            "dataset_split": self.dataset_split,
            "dataset_revision": self.dataset_revision,
            "source_content_hash": self.source_content_hash,
            "source_task_identities": list(self.source_task_identities),
            "task_identities": list(self.task_identities),
        }
        return payload

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_TASK_SET,
            payload=self.identity_payload(),
        )


@dataclass(frozen=True, slots=True)
class RepeatProvenanceRow:
    """Internal input used to materialize a Repeat Plan."""

    task_identity: str
    repeat_index: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.repeat_index) is not int:
            raise TypeError("repeat_index must be an integer")
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")
        if self.seed is not None and type(self.seed) is not int:
            raise TypeError("repeat seed must be an integer")


class RepeatId(FrozenModel):
    """The stable identity and optional RNG data for one Repeat slot."""

    repeat_plan_identity: str
    task_identity: str
    index: StrictInt
    rng_seed: StrictInt | None = None

    @field_validator("repeat_plan_identity")
    @classmethod
    def validate_plan_identity(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(
                "repeat plan identity must be a lowercase SHA-256"
            )
        return value

    @field_validator("index")
    @classmethod
    def validate_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("repeat index must be non-negative")
        return value

    def identity_payload(self) -> dict[str, object]:
        return {
            "repeat_plan_identity": self.repeat_plan_identity,
            "task_identity": self.task_identity,
            "index": self.index,
            "rng_seed": self.rng_seed,
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_REPEAT_ID,
            payload=self.identity_payload(),
        )


class Repeat(FrozenModel):
    """One deliberate independent observation, distinct from a retry."""

    repeat_id: RepeatId


def sample_identity_for(
    *,
    sampling_config_identity: str,
    repeat_id: RepeatId,
    ordinal: int,
    task_identity: str,
) -> SampleIdentity:
    if repeat_id.task_identity != task_identity:
        raise ValueError(
            "sample task identity must match its repeat task identity"
        )
    repeat_identity = repeat_id.identity_hash()
    return SampleIdentity(
        sampling_config_identity=sampling_config_identity,
        repeat_identity=repeat_identity,
        ordinal=ordinal,
        task_identity=task_identity,
        identity_hash=sample_identity_hash(
            sampling_config_identity=sampling_config_identity,
            repeat_identity=repeat_identity,
            ordinal=ordinal,
            task_identity=task_identity,
        ),
    )


class RepeatPlan(FrozenModel):
    """A deterministic task-major plan of Repeat slots."""

    plan_id: str
    version: str
    task_identities: tuple[str, ...]
    repeat_count: StrictInt
    seeds: tuple[tuple[str, StrictInt], ...] = ()

    @field_validator("repeat_count")
    @classmethod
    def validate_repeat_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repeat_count must be at least 1")
        return value

    @model_validator(mode="after")
    def validate_slots(self) -> Self:
        if not self.task_identities:
            raise ValueError("task_identities must not be empty")
        if len(set(self.task_identities)) != len(self.task_identities):
            raise ValueError("task_identities must be unique")
        valid_keys = {
            f"{task_identity}#{index}"
            for task_identity in self.task_identities
            for index in range(self.repeat_count)
        }
        seed_keys = [key for key, _seed in self.seeds]
        if len(seed_keys) != len(set(seed_keys)):
            raise ValueError("repeat seed keys must be unique")
        unknown = set(seed_keys) - valid_keys
        if unknown:
            raise ValueError(
                "repeat seeds reference unknown slots: "
                + ", ".join(sorted(unknown))
            )
        task_major_position = {
            f"{task_identity}#{index}": position
            for position, (task_identity, index) in enumerate(
                (
                    (task_identity, index)
                    for task_identity in self.task_identities
                    for index in range(self.repeat_count)
                )
            )
        }
        canonical_seed_keys = sorted(
            seed_keys,
            key=task_major_position.__getitem__,
        )
        if seed_keys != canonical_seed_keys:
            raise ValueError(
                "repeat seeds must use canonical task-major order"
            )
        return self

    def repeats(self) -> tuple[Repeat, ...]:
        seeds = dict(self.seeds)
        repeat_plan_identity = self.identity_hash()
        return tuple(
            Repeat(
                repeat_id=RepeatId(
                    repeat_plan_identity=repeat_plan_identity,
                    task_identity=task_identity,
                    index=index,
                    rng_seed=seeds.get(f"{task_identity}#{index}"),
                )
            )
            for task_identity in self.task_identities
            for index in range(self.repeat_count)
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "task_identities": list(self.task_identities),
            "repeat_count": self.repeat_count,
            "seeds": [list(seed) for seed in self.seeds],
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_REPEAT_PLAN,
            payload=self.identity_payload(),
        )


def repeat_plan_from_provenance(
    rows: tuple[RepeatProvenanceRow, ...],
    *,
    plan_id: str,
    version: str,
) -> RepeatPlan:
    """Materialize a uniform, contiguous plan from provenance rows."""

    task_order: list[str] = []
    indices_by_task: dict[str, set[int]] = {}
    seeds: dict[tuple[str, int], int] = {}
    for row in rows:
        if row.task_identity not in indices_by_task:
            task_order.append(row.task_identity)
            indices_by_task[row.task_identity] = set()
        indices = indices_by_task[row.task_identity]
        if row.repeat_index in indices:
            raise ValueError(
                "duplicate (task_identity, repeat_index) in provenance rows"
            )
        indices.add(row.repeat_index)
        if row.seed is not None:
            seeds[(row.task_identity, row.repeat_index)] = row.seed

    if not task_order:
        raise ValueError("provenance rows are empty")
    counts = {len(indices) for indices in indices_by_task.values()}
    if len(counts) != 1:
        raise ValueError(
            "every task must have the same number of repeat slots"
        )
    repeat_count = counts.pop()
    for task_identity, indices in indices_by_task.items():
        if indices != set(range(repeat_count)):
            raise ValueError(
                f"task {task_identity!r} repeat indices are not contiguous "
                f"0..{repeat_count - 1}"
            )

    seed_pairs = tuple(
        (f"{task_identity}#{index}", seeds[(task_identity, index)])
        for task_identity in task_order
        for index in range(repeat_count)
        if (task_identity, index) in seeds
    )
    return RepeatPlan(
        plan_id=plan_id,
        version=version,
        task_identities=tuple(task_order),
        repeat_count=repeat_count,
        seeds=seed_pairs,
    )


__all__ = [
    "Repeat",
    "RepeatId",
    "RepeatPlan",
    "RepeatProvenanceRow",
    "SampleIdentity",
    "TaskSet",
    "humaneval_source_content_hash",
    "humaneval_task_identity",
    "humaneval_task_identity_payload",
    "repeat_plan_from_provenance",
    "sample_identity_for",
]
