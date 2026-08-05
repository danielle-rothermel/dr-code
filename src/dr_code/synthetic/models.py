from __future__ import annotations

from dr_code.core.models import FrozenModel
from dr_code.synthetic.names import SAMPLE_ID_SEP
from dr_code.trace import ComponentSetting


class CorruptedSample(FrozenModel):
    corrupted_source: str


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
    generation_seed: int
    recipe: RecipeCoordinate


class SyntheticSample(FrozenModel):
    sample_id: str
    coordinate: SyntheticSampleCoordinate
    ground_truth_source: str
    corrupted_source: str

    @classmethod
    def make_id(
        cls,
        coordinate: SyntheticSampleCoordinate,
    ) -> str:
        return SAMPLE_ID_SEP.join(
            (
                coordinate.humaneval_task_id,
                f"{coordinate.recipe.recipe_name}@{coordinate.recipe.version}",
                str(coordinate.generation_seed),
            )
        )
