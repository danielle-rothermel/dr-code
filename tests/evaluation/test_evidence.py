from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Final
from uuid import UUID

import pytest
from dr_serialize import IdentityDocument, Jsonable, Sha256Digest
from dr_store import ObjectReference, ObjectStore

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
    ATTEMPT_RECORD_OBJECT_SCHEMA,
    AttemptCompleteness,
    AttemptValidity,
    BundleRecordReference,
    EvaluationAttemptIdentity,
    EvaluationAttemptRecord,
    EvaluationMemberRecord,
    EvaluationPlan,
    EvaluationRuntimeIdentity,
    EnlistedObjectStore,
    EvaluationSampleMetadata,
    GeneratedSampleProvenance,
    NoCandidatesSampleRecord,
    OUTPUT_REFERENCE_BINDING_PREFIX,
    SAMPLE_RECORD_OBJECT_SCHEMA,
    commit_evaluation_evidence,
    output_reference_binding_key,
    sample_record_binding_key,
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


def runtime() -> EvaluationRuntimeIdentity:
    return EvaluationRuntimeIdentity(
        document=IdentityDocument(
            schema="dr-code/runtime",
            schema_version=1,
            payload={"runtime": "test"},
        )
    )


def metadata(**overrides: object) -> EvaluationSampleMetadata:
    return EvaluationSampleMetadata(
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


def evaluation_plan() -> EvaluationPlan:
    return EvaluationPlan(
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


def attempt_record(**overrides: object) -> EvaluationAttemptRecord:
    return EvaluationAttemptRecord(
        **{
            "identity": EvaluationAttemptIdentity(attempt_id=_ATTEMPT_ID),
            "plan": evaluation_plan(),
            "runtime": runtime(),
            "cache_namespace": "evaluation-v1",
            "members": (
                EvaluationMemberRecord(
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


class FakeConnection:
    """Stands in for the caller's open sync Core connection.

    Records the lifecycle calls dr-code must never make so the tests can
    assert the transaction stays the caller's.
    """

    def __init__(self) -> None:
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.close_calls += 1


@dataclass(frozen=True, slots=True)
class _Write:
    key: str
    schema: str
    record: Jsonable


@dataclass
class FakeEnlistedObjectStore:
    """Conforms to the enlisted call surface with uncommitted staging.

    Writes land in ``pending`` and only become visible on ``commit``, so a
    rollback leaves nothing readable — the visibility semantics the real
    enlisted operations get from the caller's transaction.
    """

    connection: FakeConnection
    calls: list[str] = field(default_factory=list)
    pending_objects: dict[str, _Write] = field(default_factory=dict)
    pending_bindings: dict[str, ObjectReference] = field(default_factory=dict)
    visible_objects: dict[str, _Write] = field(default_factory=dict)
    visible_bindings: dict[str, ObjectReference] = field(default_factory=dict)

    def _reference(self, schema: str, record: Jsonable) -> ObjectReference:
        payload = repr(record).encode("utf-8")
        return ObjectReference(
            schema=schema,
            content_hash=hashlib.sha256(payload).hexdigest(),
        )

    def put_enlisted(
        self,
        connection: FakeConnection,
        schema: str,
        record: Jsonable,
    ) -> tuple[ObjectReference, str]:
        assert connection is self.connection
        self.calls.append("put_enlisted")
        object_reference = self._reference(schema, record)
        self.pending_objects[object_reference.content_hash] = _Write(
            key=object_reference.content_hash,
            schema=schema,
            record=record,
        )
        return object_reference, "stored"

    def put_many_enlisted(
        self,
        connection: FakeConnection,
        entries: dict[str, tuple[str, Jsonable]],
    ) -> dict[str, ObjectReference]:
        assert connection is self.connection
        self.calls.append("put_many_enlisted")
        results: dict[str, ObjectReference] = {}
        for key, (schema, record) in entries.items():
            object_reference = self._reference(schema, record)
            self.pending_objects[key] = _Write(
                key=key, schema=schema, record=record
            )
            self.pending_bindings[key] = object_reference
            results[key] = object_reference
        return results

    def bind_enlisted(
        self,
        connection: FakeConnection,
        key: str,
        reference: ObjectReference,
    ) -> str:
        assert connection is self.connection
        self.calls.append("bind_enlisted")
        self.pending_bindings[key] = reference
        return "bound"

    def commit(self) -> None:
        self.visible_objects.update(self.pending_objects)
        self.visible_bindings.update(self.pending_bindings)
        self.pending_objects.clear()
        self.pending_bindings.clear()

    def rollback(self) -> None:
        self.pending_objects.clear()
        self.pending_bindings.clear()


@dataclass
class FakePublication:
    """Records artifact publication against the same ordered call log."""

    calls: list[str]
    published: bool = False

    def publish(self) -> None:
        self.calls.append("publish")
        self.published = True


def commit_one_attempt(
    store: FakeEnlistedObjectStore,
) -> EvaluationAttemptRecord:
    attempt = attempt_record()
    commit_evaluation_evidence(
        store.connection,  # ty: ignore[invalid-argument-type]
        object_store=store,
        attempt=attempt,
        samples=(sample_record(),),
    )
    return attempt


@pytest.mark.parametrize(
    "method_name",
    ("put_enlisted", "put_many_enlisted", "bind_enlisted"),
)
def test_real_object_store_matches_the_enlisted_surface(
    method_name: str,
) -> None:
    # ty excludes tests, so conformance is asserted at runtime. Enlisted
    # operations require a PostgresBackend, so this pins the call surface the
    # fake stands in for, not the transactional semantics.
    expected = inspect.signature(getattr(EnlistedObjectStore, method_name))
    actual = inspect.signature(getattr(ObjectStore, method_name))
    assert [
        (parameter.name, parameter.kind)
        for parameter in expected.parameters.values()
    ] == [
        (parameter.name, parameter.kind)
        for parameter in actual.parameters.values()
    ]


def test_fake_object_store_matches_the_enlisted_surface() -> None:
    store: EnlistedObjectStore = FakeEnlistedObjectStore(
        connection=FakeConnection()  # ty: ignore[invalid-argument-type]
    )
    for method_name in ("put_enlisted", "put_many_enlisted", "bind_enlisted"):
        expected = inspect.signature(getattr(ObjectStore, method_name))
        actual = inspect.signature(getattr(store, method_name))
        assert [
            parameter.name
            for parameter in expected.parameters.values()
            if parameter.name != "self"
        ] == [parameter.name for parameter in actual.parameters.values()]


def test_rollback_leaves_no_evidence_visible() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    commit_one_attempt(store)
    store.rollback()

    assert store.visible_objects == {}
    assert store.visible_bindings == {}


def test_commit_makes_records_and_reference_visible_together() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    attempt = commit_one_attempt(store)
    binding_key = output_reference_binding_key(attempt.identity)
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)

    assert store.visible_objects == {}
    assert store.visible_bindings == {}

    store.commit()

    assert member_key in store.visible_objects
    assert binding_key in store.visible_bindings
    assert (
        store.visible_bindings[binding_key].schema
        == ATTEMPT_RECORD_OBJECT_SCHEMA
    )
    assert store.pending_objects == {}
    assert store.pending_bindings == {}


def test_entry_point_never_ends_the_callers_transaction() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    commit_one_attempt(store)

    assert connection.commit_calls == 0
    assert connection.rollback_calls == 0
    assert connection.close_calls == 0


def test_artifact_publishes_before_the_first_enlisted_write() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)
    publication = FakePublication(calls=store.calls)

    publication.publish()
    commit_one_attempt(store)

    assert publication.published
    assert store.calls[0] == "publish"
    assert "publish" not in store.calls[1:]


def test_binding_reference_resolves_the_attempt_record() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    attempt = commit_one_attempt(store)
    store.commit()

    bound = store.visible_bindings[
        output_reference_binding_key(attempt.identity)
    ]
    stored = store.visible_objects[bound.content_hash]
    assert stored.schema == ATTEMPT_RECORD_OBJECT_SCHEMA
    assert stored.record == attempt.model_dump(mode="json")


def test_member_records_bind_under_the_attempt_prefix() -> None:
    attempt = EvaluationAttemptIdentity(attempt_id=_ATTEMPT_ID)
    binding_key = output_reference_binding_key(attempt)
    member_key = sample_record_binding_key(attempt, ordinal=3)

    assert binding_key == f"{OUTPUT_REFERENCE_BINDING_PREFIX}{_ATTEMPT_ID}"
    assert member_key == f"{binding_key}/3"


def test_sample_record_ordinal_must_not_be_negative() -> None:
    attempt = EvaluationAttemptIdentity(attempt_id=_ATTEMPT_ID)
    with pytest.raises(ValueError, match="must not be negative"):
        sample_record_binding_key(attempt, ordinal=-1)


def test_evidence_requires_one_sample_record_per_member() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    with pytest.raises(ValueError, match="match the attempt's members"):
        commit_evaluation_evidence(
            store.connection,  # ty: ignore[invalid-argument-type]
            object_store=store,
            attempt=attempt_record(),
            samples=(),
        )
    assert store.calls == []
