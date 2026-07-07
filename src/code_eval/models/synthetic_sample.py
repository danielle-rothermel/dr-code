"""SyntheticSample — one row of the test corpus."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import SAMPLE_ID_SEP


class SyntheticSample(FrozenModel):
    """One row of the synthetic dataset.

    A sample combines a HumanEvalPlus task's canonical ground-truth source
    with the output of one named corruption recipe applied to it. The
    `expected_recovery_steps` field is the contract the validator must
    satisfy.
    """

    sample_id: str
    humaneval_task_id: str
    recipe_name: str
    ground_truth_source: str
    corrupted_source: str
    expected_recovery_steps: frozenset[str]
    #: Ordered subsequence of step names we expect to see in the
    #: validator's `extractor_path` for the surviving candidate.
    expected_extractor_path_contains: tuple[str, ...] = ()
    dataset_version: str

    @classmethod
    def make_id(cls, humaneval_task_id: str, recipe_name: str) -> str:
        """Canonical sample id format."""
        return f"{humaneval_task_id}{SAMPLE_ID_SEP}{recipe_name}"
