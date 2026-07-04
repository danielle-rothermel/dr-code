"""Shared helper for ruff-subprocess-based normalizers (L2/L3/L4)."""

from __future__ import annotations

from typing import Final

from code_eval.models.diagnostic import Diagnostic
from code_eval.models.normalized_form import NormalizedForm
from code_eval.names import (
    DiagnosticSeverity,
    DiagnosticSource,
    NormalizerName,
    ToolName,
)
from code_eval.subprocess_runner import SubprocessRunner

_STDIN_FILENAME: Final[str] = "candidate.py"


def run_ruff_format(
    runner: SubprocessRunner, source: str
) -> tuple[str, tuple[Diagnostic, ...], bool]:
    """Run `ruff format -` and return (out, diagnostics, success)."""
    res = runner.run(
        ToolName.RUFF.value,
        (
            "format",
            "--stdin-filename",
            _STDIN_FILENAME,
            "-",
        ),
        stdin_text=source,
    )
    if res.ok:
        return res.stdout, (), True
    return _failure(source, "ruff_format_failed", res.stderr or "ruff format failed")


def run_ruff_check_fix(
    runner: SubprocessRunner,
    source: str,
    *,
    unsafe: bool,
) -> tuple[str, tuple[Diagnostic, ...], bool]:
    """Run `ruff check --fix [--unsafe-fixes] -` and return (out, diagnostics, success).

    Ruff emits the fixed source on stdout when `--fix-only` is set. We
    use `--fix-only` plus `--exit-zero` so non-fatal lint findings do not
    fail the normalizer.
    """
    args: tuple[str, ...] = (
        "check",
        "--fix-only",
        "--exit-zero",
        "--stdin-filename",
        _STDIN_FILENAME,
    )
    if unsafe:
        args = (*args, "--unsafe-fixes")
    args = (*args, "-")
    res = runner.run(ToolName.RUFF.value, args, stdin_text=source)
    if res.ok:
        return res.stdout, (), True
    return _failure(
        source,
        "ruff_check_failed",
        res.stderr or "ruff check failed",
    )


def _failure(source: str, kind: str, message: str) -> tuple[str, tuple[Diagnostic, ...], bool]:
    return (
        source,
        (
            Diagnostic(
                source=DiagnosticSource.SUBPROCESS,
                severity=DiagnosticSeverity.WARNING,
                message=message,
                kind=kind,
                step=None,
            ),
        ),
        False,
    )


def make_form(
    normalizer: NormalizerName,
    source: str,
    diagnostics: tuple[Diagnostic, ...],
    duration_ms: float,
    success: bool,
    transformations: tuple[str, ...] = (),
) -> NormalizedForm:
    return NormalizedForm(
        normalizer=normalizer,
        source=source,
        transformations_applied=transformations or ((normalizer.value,) if success else ()),
        diagnostics=diagnostics,
        duration_ms=duration_ms,
        success=success,
    )
