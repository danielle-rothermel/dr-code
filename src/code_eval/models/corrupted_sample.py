"""CorruptedSample — output of one InverseTransform.apply() call."""

from __future__ import annotations

from code_eval.models.base import FrozenModel


class CorruptedSample(FrozenModel):
    """One inverse-transform application result.

    Inverse transforms are pure functions of (source, rng). The returned
    sample carries the corrupted source plus the set of recovery steps a
    well-behaved validator is expected to apply to undo this corruption.
    """

    corrupted_source: str
    #: Recovery step names (extractor / repair / normalizer names) the
    #: validator must surface in `extractor_path + repairs_applied` (or
    #: in the names of normalizers that produced an equivalent form) for
    #: this corruption to be considered correctly attributed.
    expected_recovery_steps: frozenset[str]
    #: Free-form notes about what was changed; mostly for debugging.
    notes: str = ""
