"""Drop candidates that are plain literal modules."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.inspected_filter import InspectedFilterStep
from dr_code.preprocessing.steps.representation import is_plain_literal_module
from dr_code.trace import InspectedCodeCandidate


class FilterPlainLiteral(InspectedFilterStep):
    """Drop candidates whose whole body is one bare container literal."""

    NAME: ClassVar[StepName] = StepName.FILTER_PLAIN_LITERAL
    VERSION: ClassVar[str] = "0"

    def rejection_reason(
        self, inspected: InspectedCodeCandidate
    ) -> str | None:
        if is_plain_literal_module(inspected.candidate.source):
            return "plain literal module"
        return None


__all__ = ["FilterPlainLiteral"]
