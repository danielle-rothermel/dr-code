"""CandidateSelection - selected candidate and ranking facts."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.candidate_rank import CandidateRank


class CandidateSelection(FrozenModel):
    """Deterministic selection facts for recovered valid candidates."""

    best_candidate_id: str | None = None
    best_attempt_id: str | None = None
    ranked_valid_candidates: tuple[CandidateRank, ...] = ()
