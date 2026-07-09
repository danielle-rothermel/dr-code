"""name_normalize — alpha-rename local variables to `_v0`, `_v1`, ...

Module-level names (functions, classes, top-level assignments, imports)
are preserved. Function arguments and locals are renamed positionally per
function. The renaming is deterministic for a given AST.
"""

from __future__ import annotations

import time
from typing import ClassVar

from dr_code.code_transforms import alpha_rename_locals

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import (
    DiagnosticSeverity,
    DiagnosticSource,
    NormalizerName,
)
from code_eval.normalizers.base import Normalizer


class NameNormalize(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.NAME_NORMALIZE

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        try:
            out = alpha_rename_locals(source)
            return NormalizedForm(
                normalizer=self.NAME,
                source=out,
                transformations_applied=(self.NAME.value,),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=True,
            )
        except SyntaxError as e:
            return NormalizedForm(
                normalizer=self.NAME,
                source=source,
                transformations_applied=(),
                diagnostics=(
                    Diagnostic(
                        source=DiagnosticSource.NORMALIZER,
                        severity=DiagnosticSeverity.WARNING,
                        message=str(e),
                        kind="name_normalize_failed",
                        step=self.NAME.value,
                    ),
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=False,
            )
