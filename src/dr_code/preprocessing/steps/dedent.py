"""Dedent each candidate."""

from __future__ import annotations

import textwrap
from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class Dedent(CandidateMapStep):
    """``textwrap.dedent`` each candidate.

    Definitions choose whether this step participates in candidate cleaning.
    """

    NAME: ClassVar[StepName] = StepName.DEDENT_CANDIDATES
    VERSION: ClassVar[str] = "1"

    def apply_to_candidate(self, source: str) -> str:
        return textwrap.dedent(source)


__all__ = ["Dedent"]
