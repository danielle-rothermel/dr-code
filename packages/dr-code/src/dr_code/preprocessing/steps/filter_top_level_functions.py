from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.inspected_filter import InspectedFilterStep
from dr_code.trace import InspectedCodeCandidate


class FilterTopLevelFunctions(InspectedFilterStep):
    NAME: ClassVar[StepName] = StepName.FILTER_TOP_LEVEL_FUNCTIONS
    VERSION: ClassVar[str] = "0"

    def rejection_reason(
        self, inspected: InspectedCodeCandidate
    ) -> str | None:
        if inspected.inspection.top_level_function_names:
            return None
        return "no top-level function definitions"


__all__ = ["FilterTopLevelFunctions"]
