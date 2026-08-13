from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.inspected_filter import InspectedFilterStep
from dr_code.trace import InspectedCodeCandidate


class FilterCompilable(InspectedFilterStep):
    NAME: ClassVar[StepName] = StepName.FILTER_COMPILABLE
    VERSION: ClassVar[str] = "0"

    def rejection_reason(
        self, inspected: InspectedCodeCandidate
    ) -> str | None:
        inspection = inspected.inspection
        if inspection.compiles:
            return None
        return inspection.compile_error or "candidate does not compile"


__all__ = ["FilterCompilable"]
