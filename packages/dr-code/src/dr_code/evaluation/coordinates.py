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


class SamplingPlanCoordinate(FrozenModel):
    sampling_plan_id: str
    version: str


class SamplingPlan(FrozenModel):
    """Declares how many samples each selected task is planned to receive.

    ``task_num_samples`` is positionally aligned with the plan's ordered task
    selection, so a plan declares exactly the slots its tasks occupy even
    when different tasks carry different sample counts.
    """

    coordinate: SamplingPlanCoordinate
    task_count: int
    task_num_samples: tuple[int, ...]
    seeds: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def validate_slot_structure(self) -> Self:
        if self.task_count < 1:
            raise ValueError("a sampling plan must cover at least one task")
        if len(self.task_num_samples) != self.task_count:
            raise ValueError(
                "a sampling plan needs one sample count per task: expected "
                f"{self.task_count}, got {len(self.task_num_samples)}"
            )
        if any(num_samples < 1 for num_samples in self.task_num_samples):
            raise ValueError("every task must have at least one sample")
        return self

    @model_validator(mode="after")
    def validate_seed_count(self) -> Self:
        if self.seeds is None:
            return self
        if len(self.seeds) != self.slot_count:
            raise ValueError(
                f"a seeded sampling plan needs one seed per slot: expected "
                f"{self.slot_count}, got {len(self.seeds)}"
            )
        return self

    @property
    def slot_count(self) -> int:
        return sum(self.task_num_samples)

    def num_samples_for(self, task_index: int) -> int:
        if not 0 <= task_index < self.task_count:
            raise ValueError(
                f"task index {task_index} is outside the plan's "
                f"{self.task_count} tasks"
            )
        return self.task_num_samples[task_index]

    def slot_index(self, task_index: int, sample_index: int) -> int:
        num_samples = self.num_samples_for(task_index)
        if not 0 <= sample_index < num_samples:
            raise ValueError(
                f"sample index {sample_index} is outside the {num_samples} "
                f"samples planned for task index {task_index}"
            )
        return sum(self.task_num_samples[:task_index]) + sample_index


__all__ = [
    "DatasetCoordinate",
    "SamplingPlan",
    "SamplingPlanCoordinate",
    "TaskSet",
    "TaskSetCoordinate",
]
