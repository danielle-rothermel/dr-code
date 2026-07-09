"""L1 — Strip comments and docstrings.

Docstrings: remove the first `Expr(Constant(str))` from every module,
function, and class body (per the plan, AST-walk based).

Comments: tokenize the source and drop `COMMENT` tokens before
re-emitting. Uses `tokenize` over the source string via a BytesIO so we
don't shell out.
"""

from __future__ import annotations

import io
import time
import token as token_mod
import tokenize
from typing import ClassVar

from dr_code.code_transforms import strip_docstrings

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import (
    DiagnosticSeverity,
    DiagnosticSource,
    NormalizerName,
)
from code_eval.normalizers.base import Normalizer


def _strip_comments(source: str) -> str:
    """Tokenize -> drop COMMENT tokens -> untokenize."""
    out_tokens: list[tokenize.TokenInfo] = []
    readline = io.BytesIO(source.encode("utf-8")).readline
    for tok in tokenize.tokenize(readline):
        if tok.type == token_mod.COMMENT:
            continue
        out_tokens.append(tok)
    return tokenize.untokenize(out_tokens).decode("utf-8")


def strip_comments_and_docstrings(source: str) -> str:
    """Apply both passes in order."""
    # Comments first (operates on tokens); docstrings second (operates on AST).
    return strip_docstrings(_strip_comments(source))


class L1StripCommentsDocstrings(Normalizer):
    NAME: ClassVar[NormalizerName] = (
        NormalizerName.L1_STRIP_COMMENTS_DOCSTRINGS
    )

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        try:
            out = strip_comments_and_docstrings(source)
            return NormalizedForm(
                normalizer=self.NAME,
                source=out,
                transformations_applied=(self.NAME.value,),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=True,
            )
        except (SyntaxError, tokenize.TokenError, IndentationError) as e:
            return NormalizedForm(
                normalizer=self.NAME,
                source=source,
                transformations_applied=(),
                diagnostics=(
                    Diagnostic(
                        source=DiagnosticSource.NORMALIZER,
                        severity=DiagnosticSeverity.WARNING,
                        message=str(e),
                        kind="l1_strip_failed",
                        step=self.NAME.value,
                    ),
                ),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                success=False,
            )
