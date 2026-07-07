"""CandidateRecoveryResult - recovered candidates plus recovery trace."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.candidate import Candidate
from code_eval.models.candidate_recovery_attempt import CandidateRecoveryAttempt
from code_eval.models.candidate_selection import CandidateSelection


class CandidateRecoveryResult(FrozenModel):
    """Full Candidate Recovery output for one Extraction result."""

    candidates: tuple[Candidate, ...]
    valid_candidates: tuple[Candidate, ...]
    attempts: tuple[CandidateRecoveryAttempt, ...]
    selection: CandidateSelection

    @property
    def overall_success(self) -> bool:
        """True when at least one recovered candidate passed all validators."""
        return len(self.valid_candidates) > 0

    def selected_candidate(self) -> Candidate | None:
        """Return the selected valid candidate, or ``None`` when recovery failed."""
        selected_id = self.selection.best_candidate_id
        if selected_id is None:
            return None
        return next(
            (
                candidate
                for candidate in self.valid_candidates
                if candidate.candidate_id == selected_id
            ),
            None,
        )

    def selected_source(self) -> str | None:
        """Source for :meth:`selected_candidate`, or ``None`` when recovery failed."""
        selected = self.selected_candidate()
        return selected.source if selected is not None else None
