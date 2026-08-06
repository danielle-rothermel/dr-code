from __future__ import annotations

import ast

from pydantic import StrictInt, StrictStr

from dr_code.core.models import FrozenModel
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
    raw_submission: StrictStr
    accepted_code: StrictStr | None
    candidate_ordinal: StrictInt | None = None
    candidate_count: StrictInt
    failure_code: StrictStr | None = None
    cause: StrictStr | None = None
    trace: Trace

    @property
    def succeeded(self) -> bool:
        return self.accepted_code is not None

    @property
    def accepted_tree(self) -> ast.Module | None:
        if self.accepted_code is None:
            return None
        return ast.parse(self.accepted_code)


def accept_first_surviving(
    candidates: tuple[InspectedCodeCandidate, ...],
) -> int | None:
    if not candidates:
        return None
    return 0


def humaneval_runner() -> BoundPreprocessingRunner:
    return bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)


def extract_humaneval_code(
    raw_submission: str,
    *,
    runner: BoundPreprocessingRunner | None = None,
) -> CodeExtractionResult:
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
    return CodeExtractionResult(
        raw_submission=raw_submission,
        accepted_code=accepted.candidate.source,
        candidate_ordinal=ordinal,
        candidate_count=len(candidates),
        trace=trace,
    )


__all__ = [
    "CodeExtractionResult",
    "accept_first_surviving",
    "extract_humaneval_code",
    "humaneval_runner",
]
