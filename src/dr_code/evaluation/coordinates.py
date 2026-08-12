from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel


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
    """Declares how many repeats each selected task is planned to receive.

    ``task_repeats`` is positionally aligned with the plan's ordered task
    selection, so a plan declares exactly the slots its tasks occupy even
    when different tasks carry different repeat counts.
    """

    coordinate: RepeatPlanCoordinate
    task_count: int
    task_repeats: tuple[int, ...]
    seeds: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def validate_slot_structure(self) -> Self:
        if self.task_count < 1:
            raise ValueError("a repeat plan must cover at least one task")
        if len(self.task_repeats) != self.task_count:
            raise ValueError(
                "a repeat plan needs one repeat count per task: expected "
                f"{self.task_count}, got {len(self.task_repeats)}"
            )
        if any(repeats < 1 for repeats in self.task_repeats):
            raise ValueError("every task must have at least one repeat")
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
        return sum(self.task_repeats)

    def repeats_for(self, task_position: int) -> int:
        if not 0 <= task_position < self.task_count:
            raise ValueError(
                f"task position {task_position} is outside the plan's "
                f"{self.task_count} tasks"
            )
        return self.task_repeats[task_position]

    def slot_index(self, task_position: int, repeat_index: int) -> int:
        repeats = self.repeats_for(task_position)
        if not 0 <= repeat_index < repeats:
            raise ValueError(
                f"repeat index {repeat_index} is outside the {repeats} "
                f"repeats planned for task position {task_position}"
            )
        return sum(self.task_repeats[:task_position]) + repeat_index


__all__ = [
    "DatasetCoordinate",
    "RepeatPlan",
    "RepeatPlanCoordinate",
    "TaskSet",
    "TaskSetCoordinate",
]
