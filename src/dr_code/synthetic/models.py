"""Models for synthetic corruption-corpus generation."""

from __future__ import annotations

from dr_code.models import FrozenModel
from dr_code.synthetic.names import SAMPLE_ID_SEP
from dr_code.trace import ComponentSetting


class CorruptedSample(FrozenModel):
    """One inverse-transform application result."""

    corrupted_source: str
    notes: str = ""


class CorruptionCoordinate(FrozenModel):
    """One registered corruption component in a recipe.

    ``settings`` carries the corruption's resolved tunables, so two recipe
    entries that differ only in settings are distinct coordinates. A
    corruption with no tunables carries an empty tuple.
    """

    registered_name: str
    version: str
    settings: tuple[ComponentSetting, ...] = ()


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
