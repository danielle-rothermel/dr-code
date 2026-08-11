from __future__ import annotations

import hashlib
import json
from uuid import UUID

import pytest
from dr_serialize import IdentityDocument, Sha256Digest
from pydantic import TypeAdapter, ValidationError

from _builders import (
    dataset,
    evaluation_slot,
    preprocessing_coordinate,
    sample_identity,
)
from dr_code.evaluation import (
    BundleRecordReference,
    CorpusSampleProvenance,
    EvaluationAttemptIdentity,
    EvaluationCandidateIdentity,
    EvaluationRuntimeIdentity,
    EvaluationSample,
    EvaluationSampleMetadata,
    EvaluationSampleProvenance,
    EvaluationSourceIdentity,
    GeneratedSampleProvenance,
    MaterializedEvaluationCandidate,
    StoredRecordReference,
    SyntheticSampleProvenance,
)
from dr_code.synthetic.models import (
    RecipeCoordinate,
    SyntheticSampleCoordinate,
)
from dr_code.trace import CodeArtifact, TextArtifact
from dr_store import ObjectReference

_DIGEST = Sha256Digest("a" * 64)
_CANDIDATE_SOURCE = "def f(): return 1"
_CANDIDATE_SOURCE_DIGEST = Sha256Digest(
    hashlib.sha256(_CANDIDATE_SOURCE.encode("utf-8")).hexdigest()
)


def bundle_reference() -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=0,
        record_sha256=_DIGEST,
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=1,
    )


def source_identity() -> EvaluationSourceIdentity:
    return EvaluationSourceIdentity(namespace="generator", value="run-1")


def generated_provenance() -> GeneratedSampleProvenance:
    return GeneratedSampleProvenance(
        source_identity=source_identity(),
        source_reference=bundle_reference(),
        generation_id="generation-1",
    )


def test_slot_and_sample_identities_are_distinct_models() -> None:
    slot = evaluation_slot()
    sample = sample_identity()
    assert slot.task_id == "t0"
    assert sample.sample_id == "sample-0"
    assert set(type(slot).model_fields) == {
        "task_set",
        "repeat_plan",
        "task_id",
        "repeat_index",
    }
    assert set(type(sample).model_fields) == {"sample_id"}


def test_sample_identity_and_source_identity_require_nonempty_values() -> None:
    with pytest.raises(ValidationError):
        type(sample_identity())(sample_id="")
    with pytest.raises(ValidationError):
        EvaluationSourceIdentity(namespace="", value="source")


def test_candidate_identity_nests_only_sample_identity_and_preprocessing() -> (
    None
):
    candidate = EvaluationCandidateIdentity(
        sample=sample_identity(),
        preprocessing=preprocessing_coordinate(),
        candidate_ordinal=0,
    )
    assert candidate.sample == sample_identity()
    assert "task_set" not in candidate.model_dump()


def test_attempt_identity_serializes_uuid_canonically() -> None:
    identity = EvaluationAttemptIdentity(
        attempt_id=UUID("12345678-1234-5678-1234-567812345678")
    )
    assert json.loads(identity.model_dump_json()) == {
        "attempt_id": "12345678-1234-5678-1234-567812345678"
    }


def test_runtime_identity_nests_the_dependency_document() -> None:
    document = IdentityDocument(
        schema="dr-code/runtime",
        schema_version=1,
        payload={"python": "3.13"},
    )
    assert EvaluationRuntimeIdentity(document=document).document is document


def test_all_sample_provenance_variants_round_trip_by_kind() -> None:
    stored = StoredRecordReference(
        reference=ObjectReference(
            schema="source-record", content_hash="b" * 64
        ),
        schema_version=1,
    )
    provenances = (
        generated_provenance(),
        CorpusSampleProvenance(
            source_identity=source_identity(),
            source_reference=stored,
            dataset=dataset(),
            row_id="row-1",
        ),
        SyntheticSampleProvenance(
            source_identity=source_identity(),
            source_reference=bundle_reference(),
            coordinate=SyntheticSampleCoordinate(
                humaneval_task_id="HumanEval/0",
                generation_seed=7,
                recipe=RecipeCoordinate(
                    recipe_name="clean", version="0", corruptions=()
                ),
            ),
        ),
    )
    adapter = TypeAdapter(EvaluationSampleProvenance)
    assert [item.kind for item in provenances] == [
        "generated",
        "corpus",
        "synthetic",
    ]
    for provenance in provenances:
        assert (
            adapter.validate_json(adapter.dump_json(provenance)) == provenance
        )


def test_request_sample_and_materialized_candidate_have_exact_wire_fields() -> (
    None
):
    metadata = EvaluationSampleMetadata(
        identity=sample_identity(),
        task_id="t0",
        provenance=generated_provenance(),
    )
    sample = EvaluationSample(
        metadata=metadata, raw_input=TextArtifact(text="raw")
    )
    candidate = MaterializedEvaluationCandidate(
        identity=EvaluationCandidateIdentity(
            sample=sample_identity(),
            preprocessing=preprocessing_coordinate(),
            candidate_ordinal=0,
        ),
        source=CodeArtifact(source=_CANDIDATE_SOURCE),
        source_sha256=_CANDIDATE_SOURCE_DIGEST,
    )
    assert set(type(sample).model_fields) == {
        "metadata",
        "raw_input",
        "auxiliary_artifacts",
    }
    assert sample.model_dump(mode="json")["auxiliary_artifacts"] == []
    assert set(type(candidate).model_fields) == {
        "identity",
        "source",
        "source_sha256",
    }


def test_materialized_candidate_rejects_a_mismatched_source_hash() -> None:
    with pytest.raises(
        ValidationError, match="must match the candidate source"
    ):
        MaterializedEvaluationCandidate(
            identity=EvaluationCandidateIdentity(
                sample=sample_identity(),
                preprocessing=preprocessing_coordinate(),
                candidate_ordinal=0,
            ),
            source=CodeArtifact(source=_CANDIDATE_SOURCE),
            source_sha256=_DIGEST,
        )


@pytest.mark.parametrize(
    "artifact_name",
    ("/absolute.jsonl", "../escape.jsonl", "a/../b.jsonl", "./a.jsonl"),
)
def test_bundle_references_reject_non_relative_or_unnormalized_names(
    artifact_name: str,
) -> None:
    with pytest.raises(ValidationError, match="normalized relative"):
        BundleRecordReference.model_validate(
            {
                **bundle_reference().model_dump(),
                "artifact_name": artifact_name,
            }
        )
