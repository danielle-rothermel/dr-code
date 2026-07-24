"""Repair structurally broken import lines per candidate."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.import_inference import repair_import_lines
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class RepairImportLines(CandidateMapStep):
    """Repair structurally broken import lines in each candidate.

    Wraps ``import_inference.repair_import_lines``: an import line that does
    not parse is rewritten when trimming trailing junk or closing unbalanced
    parentheses makes it parse, and dropped when neither does.
    """

    NAME: ClassVar[StepName] = StepName.REPAIR_IMPORT_LINES
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        repaired, _changed = repair_import_lines(source)
        return repaired


__all__ = ["RepairImportLines"]
