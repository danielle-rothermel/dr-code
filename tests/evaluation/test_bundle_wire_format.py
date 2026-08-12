from __future__ import annotations

import json

from dr_code.evaluation import (
    EVALUATION_BUNDLE_FORMAT,
    EVALUATION_BUNDLE_SCHEMA_VERSION,
    EVALUATION_PROJECTION_FORMAT,
    EVALUATION_PROJECTION_SCHEMA_VERSION,
    SAMPLE_RECORD_OBJECT_SCHEMA,
    EvaluationBundlePayload,
    ProjectionArtifactHeader,
    ProjectionKind,
)

from ._batch_builders import request


def test_evaluation_bundle_wire_literals_and_keys_are_golden() -> None:
    batch_request = request()
    projection = {
        "kind": "evaluation_samples",
        "definition_version": 2,
        "source_attempt": {
            "attempt_id": str(batch_request.attempt.attempt_id)
        },
        "artifact_name": "projection-evaluation-samples.jsonl",
    }
    payload = EvaluationBundlePayload.model_validate_json(
        json.dumps(
            {
                "format": "dr-code-evaluation-bundle-v1",
                "schema_version": 1,
                "attempt": {
                    "attempt_id": str(batch_request.attempt.attempt_id)
                },
                "attempt_artifact": "evaluation-attempt.json",
                "projections": [projection],
            }
        ),
        strict=True,
    )
    assert payload.model_dump(mode="json") == {
        "format": "dr-code-evaluation-bundle-v1",
        "schema_version": 1,
        "attempt": {"attempt_id": "00000000-0000-0000-0000-000000000001"},
        "attempt_artifact": "evaluation-attempt.json",
        "projections": [projection],
    }
    assert ProjectionArtifactHeader(
        source_attempt=batch_request.attempt,
        kind=ProjectionKind.EVALUATION_SAMPLES,
    ).model_dump(mode="json") == {
        "format": "dr-code-evaluation-projection-v1",
        "schema_version": 2,
        "source_attempt": {
            "attempt_id": "00000000-0000-0000-0000-000000000001"
        },
        "kind": "evaluation_samples",
        "definition_version": 2,
    }
    assert EVALUATION_BUNDLE_FORMAT == "dr-code-evaluation-bundle-v1"
    assert EVALUATION_BUNDLE_SCHEMA_VERSION == 1
    assert EVALUATION_PROJECTION_FORMAT == "dr-code-evaluation-projection-v1"
    assert EVALUATION_PROJECTION_SCHEMA_VERSION == 2
    assert SAMPLE_RECORD_OBJECT_SCHEMA == "dr-code/sample-evaluation-record-v1"
