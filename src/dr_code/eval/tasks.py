"""HumanEval task identity, Task Sets, and deliberate Repeat plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from pydantic import StrictInt, field_validator, model_validator

from dr_code.models import FrozenModel

if TYPE_CHECKING:
    from dr_code.humaneval.task import HumanEvalTask


def humaneval_task_identity(task: HumanEvalTask) -> str:
    """Name one HumanEval task by its dataset-assigned task id."""

    if not task.task_id:
        raise ValueError("a HumanEval task requires a task id")
    return task.task_id


class DatasetCoordinate(FrozenModel):
    """The declared coordinates of one dataset snapshot."""

    dataset_id: str
    dataset_split: str
    dataset_revision: str

    @model_validator(mode="after")
    def reject_empty_parts(self) -> Self:
        if not (
            self.dataset_id and self.dataset_split and self.dataset_revision
        ):
            raise ValueError("dataset coordinate parts must be non-empty")
        return self


class TaskSet(FrozenModel):
    """An ordered, concrete task manifest."""

    manifest_id: str
    version: str
    dataset: DatasetCoordinate
    source_task_identities: tuple[str, ...]
    task_identities: tuple[str, ...]

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
                "task_identities must belong to the declared source population"
            )
        return self

    def coordinate(self) -> TaskSetCoordinate:
        return TaskSetCoordinate(
            manifest_id=self.manifest_id,
            version=self.version,
        )


class TaskSetCoordinate(FrozenModel):
    """The manual semantic coordinate naming one Task Set."""

    manifest_id: str
    version: str


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


class RepeatPlanCoordinate(FrozenModel):
    """The manual semantic coordinate naming one Repeat Plan."""

    plan_id: str
    version: str


class RepeatId(FrozenModel):
    """The stable coordinate and optional RNG data for one Repeat slot."""

    repeat_plan: RepeatPlanCoordinate
    task_identity: str
    index: StrictInt
    rng_seed: StrictInt | None = None

    @field_validator("index")
    @classmethod
    def validate_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("repeat index must be non-negative")
        return value


class Repeat(FrozenModel):
    """One deliberate independent observation, distinct from a retry."""

    repeat_id: RepeatId


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

    def coordinate(self) -> RepeatPlanCoordinate:
        return RepeatPlanCoordinate(
            plan_id=self.plan_id,
            version=self.version,
        )

    def repeats(self) -> tuple[Repeat, ...]:
        seeds = dict(self.seeds)
        plan_coordinate = self.coordinate()
        return tuple(
            Repeat(
                repeat_id=RepeatId(
                    repeat_plan=plan_coordinate,
                    task_identity=task_identity,
                    index=index,
                    rng_seed=seeds.get(f"{task_identity}#{index}"),
                )
            )
            for task_identity in self.task_identities
            for index in range(self.repeat_count)
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
    "DatasetCoordinate",
    "Repeat",
    "RepeatId",
    "RepeatPlan",
    "RepeatPlanCoordinate",
    "RepeatProvenanceRow",
    "TaskSet",
    "TaskSetCoordinate",
    "humaneval_task_identity",
    "repeat_plan_from_provenance",
]
