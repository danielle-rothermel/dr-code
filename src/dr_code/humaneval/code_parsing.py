"""HumanEval's acceptance policy over the preprocessing candidate set.

Preprocessing answers "what code did this response contain?" and returns
every candidate that survived its structural filters. It deliberately does
not choose one: which surviving candidate a consumer accepts is that
consumer's policy, and HumanEval's policy lives here.

The policy is ``accept_first_surviving``: take the first candidate of the
materialized set. Preprocessing orders candidates by the representation
they were read from, most direct first, so the first survivor is the one
recovered by the least interpretation of the response.

``extract_humaneval_code`` runs the pipeline and applies the policy,
returning a ``CodeExtractionResult`` that carries the accepted source, its
ordinal in the materialized set, the whole preprocessing trace, and — when
nothing was accepted — the failure code preprocessing recorded.
"""

from __future__ import annotations

import ast

from pydantic import PrivateAttr, StrictInt, StrictStr

from dr_code.base import FrozenModel
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    BoundPreprocessingRunner,
    PreprocessingFailureCode,
    bind_preprocessing,
)
from dr_code.trace import (
    OUTPUT_KEY,
    InspectedCodeCandidate,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    is_absent,
)


class CodeExtractionResult(FrozenModel):
    """What HumanEval accepted from one response, and how it got there.

    ``accepted_code`` is ``None`` exactly when no candidate was accepted;
    ``failure_code`` then names why, using preprocessing's own vocabulary.
    ``candidate_ordinal`` indexes the materialized candidate set — the set
    after deduplication and filtering — and is ``None`` when nothing was
    accepted.
    """

    raw_submission: StrictStr
    accepted_code: StrictStr | None
    candidate_ordinal: StrictInt | None = None
    candidate_count: StrictInt
    failure_code: StrictStr | None = None
    cause: StrictStr | None = None
    trace: Trace

    _accepted_tree: ast.Module | None = PrivateAttr(default=None)

    @property
    def succeeded(self) -> bool:
        """True when a candidate was accepted."""
        return self.accepted_code is not None

    @property
    def accepted_tree(self) -> ast.Module | None:
        """The parsed module of the accepted source, when there is one.

        Reparsed from the accepted candidate rather than carried in the
        trace: the trace records structural *facts* about a source, not the
        tree itself, so a derived view is recomputed where it is needed.
        """
        return self._accepted_tree


def accept_first_surviving(
    candidates: tuple[InspectedCodeCandidate, ...],
) -> int | None:
    """HumanEval's acceptance policy: the first surviving candidate.

    Returns the accepted candidate's ordinal in ``candidates``, or ``None``
    when the set is empty. The policy is deliberately the simplest one that
    is truthful about what preprocessing guarantees: every element of the
    set has already passed every structural filter, so there is no
    remaining structural ground on which to prefer a later one over an
    earlier one, and set order already runs most-direct-first.
    """
    if not candidates:
        return None
    return 0


def humaneval_runner() -> BoundPreprocessingRunner:
    """Bind the definition HumanEval extracts with."""
    return bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)


def extract_humaneval_code(
    raw_submission: str,
    *,
    runner: BoundPreprocessingRunner | None = None,
) -> CodeExtractionResult:
    """Run preprocessing over one response and apply HumanEval's policy.

    Pass ``runner`` to reuse one binding across many responses; omitting it
    binds the registered definition per call.
    """
    if not isinstance(raw_submission, str):
        raise TypeError("raw_submission must be str")
    bound = runner if runner is not None else humaneval_runner()
    trace = bound.run(TextArtifact(text=raw_submission))
    output = trace.value(OUTPUT_KEY)

    if is_absent(output):
        return CodeExtractionResult(
            raw_submission=raw_submission,
            accepted_code=None,
            candidate_count=0,
            failure_code=output.failure_code,
            cause=output.cause,
            trace=trace,
        )

    assert isinstance(output, InspectedCodeCandidateSetArtifact)
    candidates = output.candidates
    ordinal = accept_first_surviving(candidates)
    if ordinal is None:
        return CodeExtractionResult(
            raw_submission=raw_submission,
            accepted_code=None,
            candidate_count=0,
            failure_code=(
                PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING.value
            ),
            cause="no candidate survived filtering",
            trace=trace,
        )

    accepted = candidates[ordinal]
    result = CodeExtractionResult(
        raw_submission=raw_submission,
        accepted_code=accepted.candidate.source,
        candidate_ordinal=ordinal,
        candidate_count=len(candidates),
        trace=trace,
    )
    result._accepted_tree = ast.parse(accepted.candidate.source)
    return result


__all__ = [
    "CodeExtractionResult",
    "accept_first_surviving",
    "extract_humaneval_code",
    "humaneval_runner",
]
