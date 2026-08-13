from __future__ import annotations

from dr_code.core.models import FrozenModel
from dr_code.trace import ComponentSetting


class CorruptionCoordinate(FrozenModel):
    registered_name: str
    version: str
    settings: tuple[ComponentSetting, ...] = ()


class RecipeCoordinate(FrozenModel):
    recipe_name: str
    version: str
    corruptions: tuple[CorruptionCoordinate, ...]


class SyntheticSampleCoordinate(FrozenModel):
    humaneval_task_id: str
    ground_truth_source_sha256: str
    generation_seed: int
    recipe: RecipeCoordinate


__all__ = [
    "CorruptionCoordinate",
    "RecipeCoordinate",
    "SyntheticSampleCoordinate",
]
