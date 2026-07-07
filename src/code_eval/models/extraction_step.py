"""ExtractionStep — one entry in the ordered extraction log.

Every extractor invocation produces one or more `ExtractionStep`s. The
ordered tuple of these on `ValidationResult.extraction_log` is the
ground-truth record of what the pipeline did, in order.
"""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import ExtractorName


class ExtractionStep(FrozenModel):
    """A single step in the extraction phase."""

    extractor: ExtractorName
    #: How many candidates this extractor produced on this invocation.
    candidates_produced: int
    #: True if the extractor surfaced at least one candidate that later
    #: passed validation. Filled in after validation completes.
    yielded_valid_candidate: bool = False
    #: Free-form short notes (e.g. "matched 2 fences", "no anchors found").
    notes: str = ""
