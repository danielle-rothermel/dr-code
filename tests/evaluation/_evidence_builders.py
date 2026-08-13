"""Attempt and sample record builders shared by the evidence test modules."""

from __future__ import annotations

from typing import Final
from uuid import UUID

from dr_serialize import IdentityDocument, Sha256Digest

from _builders import (
    evaluation_slot,
    policy,
    procedure,
    record_identity,
    sample_identity,
    sampling_plan,
    task_set,
)
from dr_code.evaluation import (
    SAMPLE_RECORD_OBJECT_SCHEMA,
    AttemptCompleteness,
    AttemptValidity,
    BundleRecordReference,
    EvalAttemptIdentity,
    EvalAttemptRecord,
    EvalMemberRecord,
    EvalPlan,
    EvalRuntimeIdentity,
    EvalSampleMetadata,
    GeneratedSampleProvenance,
    NoCandidatesSampleRecord,
)
from dr_code.trace import CodeArtifact, SerializedTrace, TextArtifact

_DIGEST: Final = Sha256Digest("a" * 64)
_ATTEMPT_ID: Final = UUID(int=2)


def reference(index: int = 0) -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=index,
        record_sha256=_DIGEST,
        schema=SAMPLE_RECORD_OBJECT_SCHEMA,
        schema_version=1,
    )


def runtime() -> EvalRuntimeIdentity:
    return EvalRuntimeIdentity(
        document=IdentityDocument(
            schema="dr-code/runtime",
            schema_version=1,
            payload={"runtime": "test"},
        )
    )


def metadata(**overrides: object) -> EvalSampleMetadata:
    return EvalSampleMetadata(
        **{
            "identity": sample_identity(),
            "task_id": "t0",
            "provenance": GeneratedSampleProvenance(
                source_identity={"namespace": "generator", "value": "run-1"},
                source_reference=reference(),
                generation_id="generation-1",
            ),
            **overrides,
        }
    )


def trace() -> SerializedTrace:
    return SerializedTrace(
        schema_version=3,
        producer=record_identity().producer,
        values={
            "input": TextArtifact(text="raw input"),
            "output": CodeArtifact(source="def f(): return 1"),
        },
    )


def evaluation_plan() -> EvalPlan:
    return EvalPlan(
        plan_id="plan",
        version="1",
        task_set=task_set(),
        sampling_plan=sampling_plan(),
        procedure=procedure(),
        aggregation=policy(),
    )


def sample_record() -> NoCandidatesSampleRecord:
    return NoCandidatesSampleRecord(
        slot=evaluation_slot(),
        sample=metadata(),
        trace=trace(),
    )


def attempt_record(**overrides: object) -> EvalAttemptRecord:
    return EvalAttemptRecord(
        **{
            "identity": EvalAttemptIdentity(attempt_id=_ATTEMPT_ID),
            "plan": evaluation_plan(),
            "runtime": runtime(),
            "cache_namespace": "evaluation-v1",
            "members": (
                EvalMemberRecord(
                    slot=evaluation_slot(),
                    sample=sample_identity(),
                    record=reference(),
                ),
            ),
            "completeness": AttemptCompleteness.COMPLETE,
            "validity": AttemptValidity.VALID,
            "limit_exhaustion": None,
            "replay": None,
            **overrides,
        }
    )
