"""L2 — Ruff format normalizer.

Pipes the source through `ruff format --stdin-filename candidate.py -`.
"""

from __future__ import annotations

import time
from typing import ClassVar

from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import NormalizerName
from code_eval.normalizers._ruff_runner import make_form, run_ruff_format
from code_eval.normalizers.base import Normalizer
from code_eval.subprocess_runner import SubprocessRunner


class L2RuffFormat(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.L2_RUFF_FORMAT

    def __init__(self, runner: SubprocessRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        out, diags, ok = run_ruff_format(self._runner, source)
        return make_form(
            self.NAME,
            out,
            diags,
            (time.perf_counter() - start) * 1000.0,
            ok,
        )
