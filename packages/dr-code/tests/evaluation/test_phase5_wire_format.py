from __future__ import annotations

import json
from uuid import UUID

import dr_code.evaluation as evaluation
from dr_code.evaluation import (
    ComparableProjectionComparison,
    ComparisonStatus,
    EvalAttemptId,
    PreprocessMode,
    ProjectionKind,
    ProjectionNotComparable,
    ReplayMode,
    ReplayReady,
    ReplaySource,
    ReplayUnavailable,
    StructuralRecordComparison,
    StructuralMemberId,
)

from ._batch_builders import request
from .test_record_models import reference


def test_replay_preflight_wire_keys_and_discriminators_are_exact() -> None:
    source = ReplaySource(
        attempt=EvalAttemptId(attempt_id=UUID(int=2)),
        mode=ReplayMode.SAMPLES,
    )
    unavailable = json.loads(
        ReplayUnavailable(
            source=source, reason="unsupported"
        ).model_dump_json()
    )
    ready = json.loads(
        ReplayReady(
            source=source,
            request=request(preprocess_mode=PreprocessMode.IN_PROCESS),
        ).model_dump_json()
    )

    assert list(unavailable) == ["kind", "source", "reason"]
    assert unavailable["kind"] == "unavailable"
    assert list(ready) == ["kind", "source", "request"]
    assert ready["kind"] == "ready"


def test_comparison_wire_keys_and_discriminators_are_exact() -> None:
    structural = json.loads(
        StructuralRecordComparison(
            identity=StructuralMemberId(
                slot=request(preprocess_mode=PreprocessMode.IN_PROCESS)
                .inputs[0]
                .slot,
                sample=request(preprocess_mode=PreprocessMode.IN_PROCESS)
                .inputs[0]
                .data.sample.metadata.identity,
            ),
            left=reference(),
            right=reference(1),
            sample=ComparisonStatus.UNCHANGED,
            trace=ComparisonStatus.CHANGED,
            candidates=ComparisonStatus.UNCHANGED,
            metrics=ComparisonStatus.CHANGED,
        ).model_dump_json()
    )
    comparable = json.loads(
        ComparableProjectionComparison(
            projection=ProjectionKind.EVAL_SAMPLES,
            population=2,
            available_denominator=1,
            changed=1,
        ).model_dump_json()
    )
    unavailable = json.loads(
        ProjectionNotComparable(
            projection=ProjectionKind.SCORES,
            left_definition_version=1,
            right_definition_version=2,
            reason="mismatch",
        ).model_dump_json()
    )

    assert list(structural) == [
        "identity",
        "left",
        "right",
        "sample",
        "trace",
        "candidates",
        "metrics",
    ]
    assert structural["trace"] == "changed"
    assert list(comparable) == [
        "kind",
        "projection",
        "definition_version",
        "population",
        "available_denominator",
        "changed",
    ]
    assert comparable["kind"] == "comparable"
    assert list(unavailable) == [
        "kind",
        "projection",
        "left_definition_version",
        "right_definition_version",
        "reason",
    ]
    assert unavailable["kind"] == "not_comparable"


def test_phase5_public_symbols_are_exported() -> None:
    for name in (
        "ReplayUnavailable",
        "ReplayReady",
        "ReplayPreflight",
        "preflight_replay",
        "replay_eval_attempt",
        "EvalEvidenceResolver",
        "ComparisonStatus",
        "StructuralRecordComparison",
        "StructuralMemberId",
        "ComparableProjectionComparison",
        "ProjectionNotComparable",
        "ProjectionComparison",
        "StructuralEvalComparison",
        "compare_eval_attempts",
    ):
        assert name in evaluation.__all__
        assert getattr(evaluation, name) is not None
