"""annotation_strip — drop type annotations from defs and assignments.

`def f(x: int) -> str:` becomes `def f(x):`. `x: int = 1` becomes `x = 1`.
"""

from __future__ import annotations

import time
from typing import ClassVar

from dr_code.code_transforms import strip_type_annotations

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import (
    DiagnosticSeverity,
    DiagnosticSource,
    NormalizerName,
)
from code_eval.normalizers.base import Normalizer


class AnnotationStrip(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.ANNOTATION_STRIP

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        try:
            out = strip_type_annotations(source)
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
                        kind="annotation_strip_failed",
                        step=self.NAME.value,
                    ),
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=False,
            )
