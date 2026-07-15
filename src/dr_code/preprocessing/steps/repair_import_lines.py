"""Repair structurally broken import lines per candidate."""

from __future__ import annotations

from typing import ClassVar

from dr_code.humaneval.import_inference import _repair_import_lines
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class RepairImportLines(CandidateMapStep):
    """Repair structurally broken import lines in each candidate.

    Wraps ``import_inference._repair_import_lines`` — the first constituent
    of ``infer_necessary_imports``.
    """

    NAME: ClassVar[StepName] = StepName.REPAIR_IMPORT_LINES
    VERSION: ClassVar[str] = "1"

    def apply_to_candidate(self, source: str) -> str:
        repaired, _changed = _repair_import_lines(source)
        return repaired


__all__ = ["RepairImportLines"]
