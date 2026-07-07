"""Diagnostic model — a structured note emitted during the pipeline."""

from __future__ import annotations

from code_eval.models.base import FrozenModel
from code_eval.names import DiagnosticSeverity, DiagnosticSource


class Diagnostic(FrozenModel):
    """A single observation or warning produced anywhere in the pipeline.

    Diagnostics are attached to the result envelope and to individual
    `NormalizedForm`s. They are never raised as exceptions.
    """

    source: DiagnosticSource
    severity: DiagnosticSeverity
    message: str
    #: Short kind tag (e.g. "subprocess_timeout", "import_repair:dedup").
    #: Free-form to avoid an over-grown enum, but expected to be stable per
    #: emitter so downstream filtering works.
    kind: str
    #: Optional name of the pipeline step that emitted the diagnostic.
    step: str | None = None
