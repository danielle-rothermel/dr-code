"""Models for synthetic corruption-corpus generation."""

from __future__ import annotations

from dr_code.models import FrozenModel
from dr_code.synthetic.names import SAMPLE_ID_SEP


class CorruptedSample(FrozenModel):
    """One inverse-transform application result."""

    corrupted_source: str
    notes: str = ""


class SyntheticSample(FrozenModel):
    """One row of the synthetic dataset."""

    sample_id: str
    humaneval_task_id: str
    recipe_name: str
    ground_truth_source: str
    corrupted_source: str

    @classmethod
    def make_id(cls, humaneval_task_id: str, recipe_name: str) -> str:
        """Canonical sample id format."""
        return f"{humaneval_task_id}{SAMPLE_ID_SEP}{recipe_name}"
