"""Extract code candidates from text via a first-success strategy ladder.

The constituents of today's extraction ladder (``candidate_blocks`` +
``strip_markdown_wrappers`` + ``recover_escaped_python``) become an
ordered strategy tuple in settings — first-success alternatives, not a
sequence of pipeline steps. The ladder stays inside this one step; which
rungs exist is data.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import ClassVar, Final

from dr_code.text_analysis import candidate_blocks, code_like_blocks
from dr_code.text_transforms import (
    recover_escaped_python,
    strip_markdown_wrappers,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    AlternativesStep,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
    TextArtifact,
)


class ExtractionStrategy(StrEnum):
    """The ladder's constituents, individually selectable per definition."""

    FENCED_BLOCKS = "fenced_blocks"  # text_analysis.candidate_blocks
    MARKDOWN_WRAPPER = (
        "markdown_wrapper"  # strip_markdown_wrappers per block
    )
    ESCAPED_PYTHON = "escaped_python"  # recover_escaped_python, re-extract
    ESCAPED_MARKDOWN_WRAPPER = (
        "escaped_markdown_wrapper"  # unescape, then strip_markdown_wrappers
    )


#: ``(text: str) -> CodeCandidateSetArtifact | None`` — None means the
#: strategy produced no candidates.
ExtractionStrategyFn = Callable[[str], CodeCandidateSetArtifact | None]


def _to_candidate_set(
    blocks: list[str],
) -> CodeCandidateSetArtifact | None:
    """Apply ``code_like_blocks`` fan-out, drop whitespace-only blocks.

    Mirrors the old pipeline's per-block refinement (``code_like_blocks``
    filters prose blocks and splits anchored segments); returns None when
    no code-like candidate survives, so first-success ladder logic falls
    through to the next strategy.
    """
    candidates = [
        block for block in code_like_blocks(blocks) if block.strip()
    ]
    if not candidates:
        return None
    return CodeCandidateSetArtifact(candidates=tuple(candidates))


def _fenced_blocks(text: str) -> CodeCandidateSetArtifact | None:
    """Fenced blocks when present, otherwise the first unfenced block."""
    return _to_candidate_set(candidate_blocks(text))


def _markdown_wrapper(text: str) -> CodeCandidateSetArtifact | None:
    """Strip one markdown wrapper marker per line of each block."""
    stripped = [
        strip_markdown_wrappers(block) for block in candidate_blocks(text)
    ]
    return _to_candidate_set(stripped)


def _escaped_python(text: str) -> CodeCandidateSetArtifact | None:
    """Recover structurally escaped Python, then re-extract candidates."""
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return None
    return _to_candidate_set(candidate_blocks(unescaped))


def _escaped_markdown_wrapper(
    text: str,
) -> CodeCandidateSetArtifact | None:
    """Recover escaped Python, then strip a markdown wrapper per block.

    Mirrors the old pipeline's ``unescaped_markdown_wrapper_fallback``: the
    unescaped-then-plain rung (``escaped_python``) can miss code hidden
    behind blockquote/list markers, so this retries with
    ``strip_markdown_wrappers`` applied to each block.
    """
    unescaped = recover_escaped_python(text)
    if unescaped is None:
        return None
    stripped = [
        strip_markdown_wrappers(block)
        for block in candidate_blocks(unescaped)
    ]
    return _to_candidate_set(stripped)


STRATEGY_REGISTRY: dict[str, ExtractionStrategyFn] = {
    ExtractionStrategy.FENCED_BLOCKS.value: _fenced_blocks,
    ExtractionStrategy.MARKDOWN_WRAPPER.value: _markdown_wrapper,
    ExtractionStrategy.ESCAPED_PYTHON.value: _escaped_python,
    ExtractionStrategy.ESCAPED_MARKDOWN_WRAPPER.value: (
        _escaped_markdown_wrapper
    ),
}

DEFAULT_STRATEGIES: Final = (
    ExtractionStrategy.FENCED_BLOCKS,
    ExtractionStrategy.MARKDOWN_WRAPPER,
    ExtractionStrategy.ESCAPED_PYTHON,
    ExtractionStrategy.ESCAPED_MARKDOWN_WRAPPER,
)


class ExtractCandidatesSettings(StepSettings):
    """Ordered, first-success strategy tuple.

    Conservative-first by the definition author's choice; the tuple is
    part of definition identity — reordering is a new definition.
    """

    alternatives: tuple[ExtractionStrategy, ...] = DEFAULT_STRATEGIES


class ExtractCandidates(AlternativesStep):
    """Text -> CandidateSet: first-success ladder over an ordered tuple.

    Resolves ``settings.alternatives`` through ``STRATEGY_REGISTRY``, in
    order. First strategy returning a non-empty candidate set wins; its
    name is recorded as ``facts["alternative"]``. All-fail raises
    ``StepFailedError``.
    """

    NAME: ClassVar[StepName] = StepName.EXTRACT_CANDIDATES
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    Settings = ExtractCandidatesSettings

    def alternatives(
        self,
    ) -> tuple[tuple[str, Callable[[Artifact], Artifact | None]], ...]:
        return tuple(
            (
                strategy.value,
                self._adapt(STRATEGY_REGISTRY[strategy.value]),
            )
            for strategy in self.settings.alternatives
        )

    @staticmethod
    def _adapt(
        strategy_fn: ExtractionStrategyFn,
    ) -> Callable[[Artifact], Artifact | None]:
        def wrapped(artifact: Artifact) -> Artifact | None:
            assert isinstance(artifact, TextArtifact)
            return strategy_fn(artifact.text)

        return wrapped


__all__ = [
    "DEFAULT_STRATEGIES",
    "STRATEGY_REGISTRY",
    "ExtractCandidates",
    "ExtractCandidatesSettings",
    "ExtractionStrategy",
    "ExtractionStrategyFn",
]
