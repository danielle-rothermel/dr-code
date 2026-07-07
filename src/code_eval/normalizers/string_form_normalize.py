"""string_form_normalize — canonicalize f-string vs `.format` vs `%`.

Uses ruff's `UP032` rule (printf/format -> f-string) via
`ruff check --select UP032 --fix-only --unsafe-fixes -`.
"""

from __future__ import annotations

import time
from typing import ClassVar, Final

from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import NormalizerName, ToolName
from code_eval.normalizers._ruff_runner import _failure, make_form
from code_eval.normalizers.base import Normalizer
from code_eval.subprocess_runner import SubprocessRunner

_STDIN_FILENAME: Final[str] = "candidate.py"


class StringFormNormalize(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.STRING_FORM_NORMALIZE

    def __init__(self, runner: SubprocessRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        res = self._runner.run(
            ToolName.RUFF.value,
            (
                "check",
                "--select",
                "UP032,UP031",
                "--fix-only",
                "--unsafe-fixes",
                "--exit-zero",
                "--stdin-filename",
                _STDIN_FILENAME,
                "-",
            ),
            stdin_text=source,
        )
        if not res.ok:
            sourced, diags, _ = _failure(
                source, "string_form_failed", res.stderr or "ruff string-form fix failed"
            )
            return make_form(
                self.NAME, sourced, diags, (time.perf_counter() - start) * 1000.0, False
            )
        return make_form(
            self.NAME,
            res.stdout,
            (),
            (time.perf_counter() - start) * 1000.0,
            True,
        )
