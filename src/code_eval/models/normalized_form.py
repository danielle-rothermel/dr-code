"""NormalizedForm — the output of one normalizer applied to one candidate."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.models.diagnostic import Diagnostic
from code_eval.names import NormalizerName


class NormalizedForm(FrozenModel):
    """Result of running a single normalizer against a candidate's source."""

    normalizer: NormalizerName
    source: str
    transformations_applied: tuple[str, ...]
    diagnostics: tuple[Diagnostic, ...] = ()
    duration_ms: float = 0.0
    #: True if the normalizer ran successfully. False indicates an exception
    #: or subprocess failure; check diagnostics for detail.
    success: bool = True
    #: True if a cached value was used.
    from_cache: bool = False
