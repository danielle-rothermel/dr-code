from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.inspected_filter import InspectedFilterStep
from dr_code.preprocessing.steps.representation import (
    is_code_representation_assignment,
)
from dr_code.trace import InspectedCodeCandidate


class FilterCodeRepr(InspectedFilterStep):
    NAME: ClassVar[StepName] = StepName.FILTER_CODE_REPR
    VERSION: ClassVar[str] = "0"

    def rejection_reason(
        self, inspected: InspectedCodeCandidate
    ) -> str | None:
        if is_code_representation_assignment(inspected.candidate.source):
            return "code representation assignment"
        return None


__all__ = ["FilterCodeRepr"]
