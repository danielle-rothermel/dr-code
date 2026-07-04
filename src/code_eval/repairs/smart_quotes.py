"""Smart-quote repair: replace U+2018/19/1C/1D with their ASCII equivalents."""

from __future__ import annotations

from typing import ClassVar, Final

from code_eval.names import RepairName
from code_eval.repairs.base import Repair, RepairResult

_QUOTE_MAP: Final[dict[str, str]] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def fix_smart_quotes(source: str) -> tuple[str, bool]:
    """Return (repaired_source, changed?)."""
    out = source
    for smart, ascii_ in _QUOTE_MAP.items():
        out = out.replace(smart, ascii_)
    return out, out != source


class SmartQuotesRepair(Repair):
    NAME: ClassVar[RepairName] = RepairName.SMART_QUOTES

    def apply(self, source: str) -> RepairResult:
        fixed, changed = fix_smart_quotes(source)
        return RepairResult(
            source=fixed,
            applied_tags=(self.NAME.value,) if changed else (),
            changed=changed,
        )
