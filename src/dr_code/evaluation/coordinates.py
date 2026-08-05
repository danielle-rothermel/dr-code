from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel
from dr_code.trace import PreprocessingDefinitionCoordinate


class DatasetCoordinate(FrozenModel):
    dataset_id: str
    version: str


class TaskSetCoordinate(FrozenModel):
    task_set_id: str
    version: str
    dataset: DatasetCoordinate


class TaskSet(FrozenModel):
    coordinate: TaskSetCoordinate
    population: tuple[str, ...]
    selected: tuple[str, ...]

    @model_validator(mode="after")
    def validate_population(self) -> Self:
        if not self.population:
            raise ValueError("a task set population must be non-empty")
        if len(set(self.population)) != len(self.population):
            raise ValueError("task set population identities must be unique")
        return self

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.selected:
            raise ValueError("a task set selection must be non-empty")
        if len(set(self.selected)) != len(self.selected):
            raise ValueError("task set selected identities must be unique")
        positions: list[int] = []
        index = {task: at for at, task in enumerate(self.population)}
        for task in self.selected:
            if task not in index:
                raise ValueError(
                    f"selected task {task!r} is not in the task set population"
                )
            positions.append(index[task])
        if positions != sorted(positions):
            raise ValueError(
                "selected tasks must preserve population order; "
                f"{self.selected!r} is not a subsequence of the population"
            )
        return self


class RepeatPlanCoordinate(FrozenModel):
    repeat_plan_id: str
    version: str


class RepeatPlan(FrozenModel):
    coordinate: RepeatPlanCoordinate
    task_count: int
    repeats: int
    seeds: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def validate_slot_structure(self) -> Self:
        if self.task_count < 1:
            raise ValueError("a repeat plan must cover at least one task")
        if self.repeats < 1:
            raise ValueError("a repeat plan must have at least one repeat")
        return self

    @model_validator(mode="after")
    def validate_seed_count(self) -> Self:
        if self.seeds is None:
            return self
        if len(self.seeds) != self.slot_count:
            raise ValueError(
                f"a seeded repeat plan needs one seed per slot: expected "
                f"{self.slot_count}, got {len(self.seeds)}"
            )
        return self

    @property
    def slot_count(self) -> int:
        return self.task_count * self.repeats

    def slot_index(self, task_position: int, repeat_index: int) -> int:
        if not 0 <= task_position < self.task_count:
            raise ValueError(
                f"task position {task_position} is outside the plan's "
                f"{self.task_count} tasks"
            )
        if not 0 <= repeat_index < self.repeats:
            raise ValueError(
                f"repeat index {repeat_index} is outside the plan's "
                f"{self.repeats} repeats"
            )
        return task_position * self.repeats + repeat_index


class SampleCoordinate(FrozenModel):
    task_set: TaskSetCoordinate
    repeat_plan: RepeatPlanCoordinate
    task: str
    repeat_index: int

    @model_validator(mode="after")
    def validate_repeat_index(self) -> Self:
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")
        return self

    def within(self, plan: RepeatPlan) -> bool:
        return (
            self.repeat_plan == plan.coordinate
            and 0 <= self.repeat_index < plan.repeats
        )


class CandidateCoordinate(FrozenModel):
    """Assigns ordinals after exact-source deduplication and filtering by `materialize_candidate_set`."""

    sample: SampleCoordinate
    preprocessing: PreprocessingDefinitionCoordinate
    candidate_ordinal: int

    @model_validator(mode="after")
    def validate_candidate_ordinal(self) -> Self:
        if self.candidate_ordinal < 0:
            raise ValueError("candidate_ordinal must be non-negative")
        return self


__all__ = [
    "CandidateCoordinate",
    "DatasetCoordinate",
    "RepeatPlan",
    "RepeatPlanCoordinate",
    "SampleCoordinate",
    "TaskSet",
    "TaskSetCoordinate",
]
