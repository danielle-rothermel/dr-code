from __future__ import annotations

from dr_code.core.models import FrozenModel
from dr_code.evaluation.provenance import (
    CorruptionCoordinate,
    RecipeCoordinate,
    SyntheticSampleCoordinate,
)
from drc_synthetic.names import SAMPLE_ID_SEP


class CorruptedSample(FrozenModel):
    corrupted_source: str


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
                coordinate.ground_truth_source_sha256,
                f"{coordinate.recipe.recipe_name}@{coordinate.recipe.version}",
                str(coordinate.generation_seed),
            )
        )


__all__ = [
    "CorruptedSample",
    "CorruptionCoordinate",
    "RecipeCoordinate",
    "SyntheticSample",
    "SyntheticSampleCoordinate",
]
