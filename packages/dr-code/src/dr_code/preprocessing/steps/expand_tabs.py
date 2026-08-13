from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepSettings, StepOutput
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class ExpandTabsSettings(StepSettings):
    tab_width: int = 4


class ExpandTabs(Step[ExpandTabsSettings]):
    NAME: ClassVar[StepName] = StepName.EXPAND_TABS
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    Settings = ExpandTabsSettings

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        return StepOutput(
            value=TextArtifact(
                text=value.text.expandtabs(self.settings.tab_width)
            )
        )


__all__ = ["ExpandTabs", "ExpandTabsSettings"]
