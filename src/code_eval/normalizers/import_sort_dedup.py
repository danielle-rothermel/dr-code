"""import_sort_dedup — apply ruff's `I` rules and explicit dedup.

Runs `ruff check --select I --fix-only --exit-zero` and then dedupes
exact-duplicate import lines (kept first).
"""

from __future__ import annotations

import time
from typing import ClassVar, Final

from dr_code.code_transforms import dedupe_imports

from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import NormalizerName, ToolName
from code_eval.normalizers._ruff_runner import _failure, make_form
from code_eval.normalizers.base import Normalizer
from code_eval.subprocess_runner import SubprocessRunner

_STDIN_FILENAME: Final[str] = "candidate.py"


class ImportSortDedup(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.IMPORT_SORT_DEDUP

    def __init__(self, runner: SubprocessRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        res = self._runner.run(
            ToolName.RUFF.value,
            (
                "check",
                "--select",
                "I",
                "--fix-only",
                "--exit-zero",
                "--stdin-filename",
                _STDIN_FILENAME,
                "-",
            ),
            stdin_text=source,
        )
        if not res.ok:
            sourced, diags, _ = _failure(
                source, "import_sort_failed", res.stderr or "ruff I-fix failed"
            )
            return make_form(
                self.NAME,
                sourced,
                diags,
                (time.perf_counter() - start) * 1000.0,
                False,
            )

        try:
            deduped = dedupe_imports(res.stdout)
        except SyntaxError as e:
            sourced, diags, _ = _failure(
                res.stdout, "import_dedup_failed", str(e)
            )
            return make_form(
                self.NAME,
                sourced,
                diags,
                (time.perf_counter() - start) * 1000.0,
                False,
            )
        return make_form(
            self.NAME,
            deduped,
            (),
            (time.perf_counter() - start) * 1000.0,
            True,
        )
