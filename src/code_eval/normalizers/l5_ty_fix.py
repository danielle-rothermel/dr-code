"""L5 — ``ty`` type-checker normalizer.

The plan calls for layering ``ty`` fixes on L4. ``ty`` 0.0.51 (the pinned
version) is read-only — it has no ``--fix`` subcommand yet. We invoke
``ty check`` against the source piped via a temp file (ty currently has
no stdin mode), capture the diagnostics, and pass the source through
unchanged.

If ``ty`` is not on PATH at all, the normalizer is a clean no-op with a
warning diagnostic, per the plan's fallback rule.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import ClassVar

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import (
    DiagnosticSeverity,
    DiagnosticSource,
    NormalizerName,
    ToolName,
)
from code_eval.normalizers.base import Normalizer
from code_eval.subprocess_runner import SubprocessRunner


class L5TyFix(Normalizer):
    NAME: ClassVar[NormalizerName] = NormalizerName.L5_TY_FIX

    def __init__(self, runner: SubprocessRunner | None = None) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()

    def normalize(self, source: str) -> NormalizedForm:
        start = time.perf_counter()
        if shutil.which(ToolName.TY.value) is None:
            return self._noop(
                source,
                start,
                "ty not installed; L5 is a documented no-op fallback.",
                kind="ty_unavailable",
            )

        # ty 0.0.51 has no --fix and no stdin mode; write a temp file.
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "candidate.py"
            target.write_text(source, encoding="utf-8")
            res = self._runner.run(
                ToolName.TY.value,
                ("check", str(target)),
                stdin_text=None,
            )

        diags: list[Diagnostic] = []
        if not res.tool_found:
            return self._noop(
                source,
                start,
                "ty disappeared mid-run",
                kind="ty_unavailable",
            )
        if res.stderr:
            diags.append(
                Diagnostic(
                    source=DiagnosticSource.SUBPROCESS,
                    severity=DiagnosticSeverity.INFO,
                    message=res.stderr.strip(),
                    kind="ty_stderr",
                    step=self.NAME.value,
                )
            )

        # Source unchanged; ty has no auto-fix.
        return NormalizedForm(
            normalizer=self.NAME,
            source=source,
            transformations_applied=(self.NAME.value,),
            diagnostics=tuple(diags),
            duration_ms=(time.perf_counter() - start) * 1000.0,
            success=True,
        )

    def _noop(self, source: str, start: float, msg: str, *, kind: str) -> NormalizedForm:
        return NormalizedForm(
            normalizer=self.NAME,
            source=source,
            transformations_applied=(),
            diagnostics=(
                Diagnostic(
                    source=DiagnosticSource.NORMALIZER,
                    severity=DiagnosticSeverity.WARNING,
                    message=msg,
                    kind=kind,
                    step=self.NAME.value,
                ),
            ),
            duration_ms=(time.perf_counter() - start) * 1000.0,
            success=True,
        )
