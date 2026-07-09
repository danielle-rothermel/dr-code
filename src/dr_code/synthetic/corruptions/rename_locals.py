"""Alpha-rename local variable names to obscure names."""

from __future__ import annotations

import random
from typing import ClassVar

from dr_code.code_transforms import alpha_rename_locals
from dr_code.synthetic.models import CorruptedSample
from dr_code.synthetic.names import CorruptionName
from dr_code.synthetic.corruptions.base import Corruption


class RenameLocals(Corruption):
    """Rename local variables to opaque `_vN` form.

    Parameters keep their names — they are part of the signature contract.
    """

    NAME: ClassVar[CorruptionName] = CorruptionName.RENAME_LOCALS

    def apply(self, source: str, rng: random.Random) -> CorruptedSample:
        try:
            corrupted = alpha_rename_locals(source, rename_params=False)
        except SyntaxError:
            corrupted = source
        return CorruptedSample(
            corrupted_source=corrupted,
        )
