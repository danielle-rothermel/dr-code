"""Expand tabs to spaces."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepSettings, StepOutput
from dr_code.trace import ArtifactKind, TextArtifact


class ExpandTabsSettings(StepSettings):
    """Number of spaces per tab."""

    tab_width: int = 4


class ExpandTabs(Step):
    """Expand tabs to spaces using ``str.expandtabs(tab_width)``."""

    NAME: ClassVar[StepName] = StepName.EXPAND_TABS
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    Settings = ExpandTabsSettings

    def apply(self, value: TextArtifact) -> StepOutput:
        return StepOutput(
            value=TextArtifact(
                text=value.text.expandtabs(self.settings.tab_width)
            )
        )


__all__ = ["ExpandTabs", "ExpandTabsSettings"]
