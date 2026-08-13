from __future__ import annotations

import pytest

from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step, StepFailedError
from dr_code.preprocessing.steps.inspect_candidates import InspectCandidates
from dr_code.trace import (
    ArtifactKind,
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
)


def _candidate_set(*sources: str) -> CodeCandidateSetArtifact:
    return CodeCandidateSetArtifact(
        candidates=tuple(
            CodeCandidate(
                source=source,
                origins=(
                    CandidateOrigin(
                        operation=ExtractionOperation(
                            operation_name="text_segments"
                        ),
                        input_location=index,
                    ),
                ),
            )
            for index, source in enumerate(sources)
        )
    )


def _inspected(*sources: str) -> InspectedCodeCandidateSetArtifact:
    value = InspectCandidates().apply(_candidate_set(*sources)).value
    assert isinstance(value, InspectedCodeCandidateSetArtifact)
    return value


def _apply_twice(step_cls: type[Step], value) -> object:
    step = step_cls(step_cls.Settings())
    try:
        first = step.apply(value)
    except StepFailedError as exc:
        first = ("failed", exc.code, exc.cause)
    try:
        second = step.apply(value)
    except StepFailedError as exc:
        second = ("failed", exc.code, exc.cause)
    return (first, second)


def _sample_for(step_cls: type[Step]):
    if step_cls.INPUT is ArtifactKind.TEXT:
        return TextArtifact(text="```python\ndef f():\n    return 1\n```\n")
    if step_cls.INPUT is ArtifactKind.INSPECTED_CODE_CANDIDATE_SET:
        return _inspected(
            "def f():\n    return 1\n", "def g():\n    return 2\n"
        )
    return _candidate_set(
        "def f():\n    return 1\n", "def g():\n    return 2\n"
    )


@pytest.mark.parametrize("step_cls", REGISTRY.values())
def test_step_is_deterministic(step_cls: type[Step]) -> None:
    first, second = _apply_twice(step_cls, _sample_for(step_cls))
    assert first == second
