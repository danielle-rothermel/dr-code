"""ExtractionFragment — raw source emitted by one extraction heuristic."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import ExtractorName


class ExtractionFragment(FrozenModel):
    """Source plus notes before the Extraction catalog attaches Attribution."""

    source: str
    notes: str = ""
    emitted_as: ExtractorName | None = None
