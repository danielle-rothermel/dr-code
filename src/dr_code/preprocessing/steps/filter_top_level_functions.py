"""Keep only candidates that define at least one top-level function."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.inspected_filter import InspectedFilterStep
from dr_code.trace import InspectedCodeCandidate


class FilterTopLevelFunctions(InspectedFilterStep):
    """Drop candidates whose inspection records no top-level function.

    Reads ``inspection.top_level_function_names``, collected from the tree
    the candidate was inspected with. A candidate that defines no
    module-level function defines nothing a function-completion task can
    call, whatever else it may contain.
    """

    NAME: ClassVar[StepName] = StepName.FILTER_TOP_LEVEL_FUNCTIONS
    VERSION: ClassVar[str] = "0"

    def rejection_reason(
        self, inspected: InspectedCodeCandidate
    ) -> str | None:
        if inspected.inspection.top_level_function_names:
            return None
        return "no top-level function definitions"


__all__ = ["FilterTopLevelFunctions"]
