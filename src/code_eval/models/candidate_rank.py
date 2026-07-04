"""CandidateRank - selection facts for one valid candidate."""

from __future__ import annotations

from code_eval.models.base import FrozenModel


class CandidateRank(FrozenModel):
    """Tie-break facts used to select the best recovered candidate."""

    candidate_id: str
    attempt_id: str
    rank_key: tuple[int, int, int, str]
    repair_count: int
    extractor_path_length: int
    uses_text_normalize: bool
