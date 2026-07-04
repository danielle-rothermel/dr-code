"""Normalizer base class.

Normalizers consume a validated candidate source and produce a
`NormalizedForm`. They never raise — failures become diagnostics on the
returned form (`success=False`).
"""

from __future__ import annotations

from typing import ClassVar

from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import NormalizerName


class Normalizer:
    """Base contract for all normalizers."""

    NAME: ClassVar[NormalizerName]

    def normalize(self, source: str) -> NormalizedForm:
        raise NotImplementedError
