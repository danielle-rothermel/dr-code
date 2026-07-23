"""Typed code artifacts for the evaluation kernel.

Three native types sit between the preprocessing boundary and metric
extraction:

- :class:`PythonSource` — the Python Source *role* carried by
  ``TextArtifact.text``. It is a thin typed wrapper documenting that a
  string is being treated as Python Source; it does **not** guarantee
  compilation (that is the Code Artifact's job).
- :class:`CodeCandidate` — one ordered candidate with a stable position
  and lineage; the pipeline moves these, never bare strings.
- :class:`CodeCandidateSet` — an ordered set of candidates. An empty set
  is a legitimate value (zero candidates survived preprocessing), **not**
  a Preprocessing Failure.
- :class:`CodeArtifact` — Python Source that has passed a construction
  time compile check. Invalid source cannot inhabit the type.

The compile-validating :class:`CodeArtifact` here is distinct from the
trace-boundary ``dr_code.trace.CodeArtifact``, which deliberately carries
possibly-invalid intermediate source through the preprocessing pipeline.
"""

from __future__ import annotations

import ast
from typing import Self

from dr_code.models import FrozenModel
from dr_code.trace.artifacts import CandidateLineage


class CodeCompilationError(ValueError):
    """Raised when source offered to :class:`CodeArtifact` will not compile.

    Carries the originating ``SyntaxError`` so callers can surface the
    exact location without re-compiling.
    """

    def __init__(self, source: str, cause: SyntaxError) -> None:
        super().__init__(f"source does not compile: {cause}")
        self.source = source
        self.cause = cause


class PythonSource(FrozenModel):
    """The Python Source role over a text value.

    Carries the same string a ``TextArtifact.text`` would; the type only
    records that the string is being read as Python Source. No compilation
    guarantee — see :class:`CodeArtifact` for that.
    """

    text: str


class CodeCandidate(FrozenModel):
    """One ordered candidate produced by preprocessing.

    ``position`` is the stable 0-based ordinal within its
    :class:`CodeCandidateSet`; ``origin`` records the lineage step that
    emitted the candidate. Candidates are never reordered by position;
    it is identity, not a sort key.

    ``lineage`` optionally carries the full multi-origin extraction lineage
    (stable candidate id plus every :class:`CandidateOrigin` path) produced
    by the preprocessing stack. A single ``origin`` string is not sufficient
    to retain multi-origin lineage, so richer producers populate ``lineage``
    losslessly while ``origin`` keeps the simple summary for existing
    consumers. ``lineage`` defaults empty; legacy producers stay unchanged.
    """

    position: int
    source: str
    origin: str
    lineage: CandidateLineage = CandidateLineage()

    def to_python_source(self) -> PythonSource:
        return PythonSource(text=self.source)


class CodeCandidateSet(FrozenModel):
    """Ordered candidates with stable positions and shared lineage.

    An **empty** set is a valid outcome: preprocessing ran and zero
    candidates survived. That is distinct from a Preprocessing Failure
    (native ``Absent``). Callers MUST NOT treat emptiness as failure.
    """

    candidates: tuple[CodeCandidate, ...] = ()

    @classmethod
    def from_sources(
        cls,
        sources: tuple[str, ...],
        *,
        origin: str,
    ) -> Self:
        """Build a set assigning contiguous positions to ``sources``."""

        return cls(
            candidates=tuple(
                CodeCandidate(position=index, source=source, origin=origin)
                for index, source in enumerate(sources)
            )
        )

    @classmethod
    def from_lineage(
        cls,
        entries: tuple[tuple[str, str, CandidateLineage], ...],
    ) -> Self:
        """Build a set preserving each candidate's full extraction lineage.

        ``entries`` is an ordered ``(source, origin, lineage)`` sequence;
        contiguous positions are assigned in order. Unlike
        :meth:`from_sources`, this retains the multi-origin
        :class:`CandidateLineage` losslessly (decision: lossless is the bar).
        """

        return cls(
            candidates=tuple(
                CodeCandidate(
                    position=index,
                    source=source,
                    origin=origin,
                    lineage=lineage,
                )
                for index, (source, origin, lineage) in enumerate(entries)
            )
        )

    @property
    def is_empty(self) -> bool:
        return len(self.candidates) == 0

    def positions(self) -> tuple[int, ...]:
        return tuple(candidate.position for candidate in self.candidates)


class CodeArtifact(FrozenModel):
    """Python Source validated at construction: invalid source is rejected.

    ``compile(source, "<code-artifact>", "exec")`` must succeed for the
    instance to exist. The canonical value is the source string; the AST
    is a derived view recomputed on demand.
    """

    source: str

    def __init__(self, /, **data: object) -> None:
        super().__init__(**data)
        try:
            compile(self.source, "<code-artifact>", "exec")
        except SyntaxError as exc:
            raise CodeCompilationError(self.source, exc) from exc

    @classmethod
    def try_from_candidate(
        cls,
        candidate: CodeCandidate,
    ) -> CodeArtifact | None:
        """Construct from a candidate, or ``None`` if it will not compile."""

        try:
            return cls(source=candidate.source)
        except CodeCompilationError:
            return None

    def module(self) -> ast.Module:
        """Derived view: re-parsed on demand, never stored."""

        return ast.parse(self.source)


__all__ = [
    "CodeArtifact",
    "CodeCandidate",
    "CodeCandidateSet",
    "CodeCompilationError",
    "PythonSource",
]
