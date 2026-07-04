"""L4 — Ruff check --fix --unsafe-fixes, then re-format."""

from __future__ import annotations

import time
from typing import ClassVar

from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import NormalizerName
from code_eval.normalizers._ruff_runner import (
    make_form,
    run_ruff_check_fix,
    run_ruff_format,
)
from code_eval.normalizers.base import Normalizer
from code_eval.subprocess_runner import SubprocessRunner


class L4RuffFixUnsafe(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.L4_RUFF_FIX_UNSAFE

    def __init__(self, runner: SubprocessRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        fixed, diags1, ok1 = run_ruff_check_fix(self._runner, source, unsafe=True)
        if not ok1:
            return make_form(
                self.NAME,
                source,
                diags1,
                (time.perf_counter() - start) * 1000.0,
                False,
            )
        formatted, diags2, ok2 = run_ruff_format(self._runner, fixed)
        return make_form(
            self.NAME,
            formatted if ok2 else fixed,
            diags1 + diags2,
            (time.perf_counter() - start) * 1000.0,
            ok2,
        )
