"""Dedent repair.

Wraps `textwrap.dedent`. Used by the pipeline as a candidate-repair pass
for inputs that arrived from extractors that *didn't* already dedent.
"""

from __future__ import annotations

import textwrap
from typing import ClassVar

from code_eval.names import RepairName
from code_eval.repairs.base import Repair, RepairResult


def fix_dedent(source: str) -> tuple[str, bool]:
    dedented = textwrap.dedent(source)
    return dedented, dedented != source


class DedentRepair(Repair):
    NAME: ClassVar[RepairName] = RepairName.DEDENT

    def apply(self, source: str) -> RepairResult:
        dedented, changed = fix_dedent(source)
        return RepairResult(
            source=dedented,
            applied_tags=(self.NAME.value,) if changed else (),
            changed=changed,
        )
