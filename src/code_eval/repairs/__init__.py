"""Repair registry.

The pipeline applies repairs in the order they appear in `REPAIRS`,
matching the plan:

  smart_quotes -> dedent -> truncation -> imports

Each repair is applied independently (each alone) and then in
combination (all together) per the plan's "each alone, then all
together" rule.
"""

from typing import Final

from code_eval.repairs.base import Repair, RepairResult
from code_eval.repairs.dedent import DedentRepair
from code_eval.repairs.imports import ImportRepair
from code_eval.repairs.smart_quotes import SmartQuotesRepair
from code_eval.repairs.truncation import TruncationRepair

REPAIRS: Final[tuple[type[Repair], ...]] = (
    SmartQuotesRepair,
    DedentRepair,
    TruncationRepair,
    ImportRepair,
)

__all__ = [
    "REPAIRS",
    "DedentRepair",
    "ImportRepair",
    "Repair",
    "RepairResult",
    "SmartQuotesRepair",
    "TruncationRepair",
]
