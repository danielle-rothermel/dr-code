"""L0 — Canonical AST normalizer.

Round-trip `ast.parse -> ast.unparse`. Strongest canonicalization in the
stack; used as the semantic-equivalence baseline.
"""

from __future__ import annotations

import ast
import time
from typing import ClassVar

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import DiagnosticSeverity, DiagnosticSource, NormalizerName
from code_eval.normalizers.base import Normalizer


class L0CanonicalAst(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.L0_CANONICAL_AST

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        try:
            tree = ast.parse(source)
            out = ast.unparse(tree)
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
                        severity=DiagnosticSeverity.ERROR,
                        message=str(e),
                        kind="l0_parse_error",
                        step=self.NAME.value,
                    ),
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=False,
            )
