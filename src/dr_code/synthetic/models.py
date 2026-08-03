"""Models for synthetic corruption-corpus generation."""

from __future__ import annotations

from dr_code.models import FrozenModel
from dr_code.synthetic.names import SAMPLE_ID_SEP


class CorruptedSample(FrozenModel):
    """One inverse-transform application result."""

    corrupted_source: str
    notes: str = ""


class CorruptionCoordinate(FrozenModel):
    """One registered corruption component in a recipe."""

    registered_name: str
    version: str


class RecipeCoordinate(FrozenModel):
    """Complete structured coordinate for one synthetic recipe."""

    recipe_name: str
    version: str
    corruptions: tuple[CorruptionCoordinate, ...]


class SyntheticSampleCoordinate(FrozenModel):
    """Complete semantic identity for one generated synthetic sample."""

    humaneval_task_id: str
    generation_seed: int
    recipe: RecipeCoordinate


class SyntheticSample(FrozenModel):
    """One row of the synthetic dataset."""

    #: Human-readable label only. ``coordinate`` is the semantic identity.
    sample_id: str
    coordinate: SyntheticSampleCoordinate
    ground_truth_source: str
    corrupted_source: str

    @classmethod
    def make_id(
        cls,
        coordinate: SyntheticSampleCoordinate,
    ) -> str:
        """Return a concise display label, not a semantic identity."""
        return SAMPLE_ID_SEP.join(
            (
                coordinate.humaneval_task_id,
                f"{coordinate.recipe.recipe_name}@{coordinate.recipe.version}",
                str(coordinate.generation_seed),
            )
        )
