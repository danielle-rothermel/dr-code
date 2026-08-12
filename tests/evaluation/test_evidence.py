from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import pytest
from dr_serialize import Jsonable
from dr_store import (
    BindingConflictError,
    BindStatus,
    ObjectReference,
    ObjectStore,
    PutStatus,
)

from _builders import sample_identity
from _evidence_builders import (
    _ATTEMPT_ID,
    attempt_record,
    reference,
    sample_record,
)
from dr_code.evaluation import (
    ATTEMPT_RECORD_OBJECT_SCHEMA,
    EnlistedObjectStore,
    EvaluationAttemptIdentity,
    EvaluationAttemptRecord,
    EvaluationMemberRecord,
    OUTPUT_REFERENCE_BINDING_PREFIX,
    SAMPLE_RECORD_OBJECT_SCHEMA,
    commit_evaluation_evidence,
    output_reference_binding_key,
    sample_record_binding_key,
)


@dataclass(frozen=True, slots=True)
class _Write:
    key: str
    schema: str
    record: Jsonable


@dataclass
class FakeConnection:
    """Stands in for the caller's open sync Core connection.

    The staged writes live here, mirroring the real enlisted operations:
    dr-store writes through the connection it is handed, and that
    connection's ``commit``/``rollback`` decide visibility. Lifecycle calls
    are counted so the tests can assert the transaction stays the caller's.
    """

    commit_calls: int = 0
    rollback_calls: int = 0
    close_calls: int = 0
    staged_objects: dict[str, _Write] = field(default_factory=dict)
    staged_bindings: dict[str, ObjectReference] = field(default_factory=dict)
    visible_objects: dict[str, _Write] = field(default_factory=dict)
    visible_bindings: dict[str, ObjectReference] = field(default_factory=dict)

    def commit(self) -> None:
        self.commit_calls += 1
        self.visible_objects.update(self.staged_objects)
        self.visible_bindings.update(self.staged_bindings)
        self.staged_objects.clear()
        self.staged_bindings.clear()

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.staged_objects.clear()
        self.staged_bindings.clear()

    def close(self) -> None:
        self.close_calls += 1


@dataclass
class FakeEnlistedObjectStore:
    """Conforms to the enlisted call surface with real enlisted semantics.

    Every write goes through the passed connection and stays staged there
    until that connection commits. ``put_many_enlisted`` is first-writer-wins
    (an occupied key keeps its existing reference and is returned as the
    winner) and ``bind_enlisted`` raises ``BindingConflictError`` when the key
    already holds a different reference.
    """

    connection: FakeConnection
    calls: list[str] = field(default_factory=list)
    bypass_writes: int = 0

    def _bound(
        self, connection: FakeConnection, key: str
    ) -> ObjectReference | None:
        staged = connection.staged_bindings.get(key)
        if staged is not None:
            return staged
        return connection.visible_bindings.get(key)

    def _stage_object(
        self,
        connection: FakeConnection,
        *,
        key: str,
        schema: str,
        record: Jsonable,
    ) -> None:
        if connection is not self.connection:
            self.bypass_writes += 1
        connection.staged_objects[key] = _Write(
            key=key, schema=schema, record=record
        )

    def put_enlisted(
        self,
        connection: FakeConnection,
        schema: str,
        record: Jsonable,
    ) -> tuple[ObjectReference, PutStatus]:
        self.calls.append("put_enlisted")
        object_reference = ObjectReference.for_record(schema, record)
        self._stage_object(
            connection,
            key=object_reference.content_hash,
            schema=schema,
            record=record,
        )
        return object_reference, PutStatus.STORED

    def put_many_enlisted(
        self,
        connection: FakeConnection,
        entries: dict[str, tuple[str, Jsonable]],
    ) -> dict[str, ObjectReference]:
        self.calls.append("put_many_enlisted")
        winners: dict[str, ObjectReference] = {}
        for key, (schema, record) in entries.items():
            existing = self._bound(connection, key)
            if existing is not None:
                winners[key] = existing
                continue
            object_reference = ObjectReference.for_record(schema, record)
            self._stage_object(
                connection, key=key, schema=schema, record=record
            )
            connection.staged_bindings[key] = object_reference
            winners[key] = object_reference
        return winners

    def bind_enlisted(
        self,
        connection: FakeConnection,
        key: str,
        reference: ObjectReference,
    ) -> BindStatus:
        self.calls.append("bind_enlisted")
        existing = self._bound(connection, key)
        if existing is None:
            if connection is not self.connection:
                self.bypass_writes += 1
            connection.staged_bindings[key] = reference
            return BindStatus.BOUND
        if existing == reference:
            return BindStatus.IDEMPOTENT
        raise BindingConflictError(
            key=key, existing=existing, requested=reference
        )


@dataclass
class RaisingEnlistedObjectStore(FakeEnlistedObjectStore):
    """Fails one enlisted op to exercise the caller's rollback obligation."""

    fail_on: str = "put_enlisted"

    def put_enlisted(
        self,
        connection: FakeConnection,
        schema: str,
        record: Jsonable,
    ) -> tuple[ObjectReference, PutStatus]:
        if self.fail_on == "put_enlisted":
            self.calls.append("put_enlisted")
            raise _EnlistedWriteFailed(self.fail_on)
        return super().put_enlisted(connection, schema, record)

    def bind_enlisted(
        self,
        connection: FakeConnection,
        key: str,
        reference: ObjectReference,
    ) -> BindStatus:
        if self.fail_on == "bind_enlisted":
            self.calls.append("bind_enlisted")
            raise _EnlistedWriteFailed(self.fail_on)
        return super().bind_enlisted(connection, key, reference)


class _EnlistedWriteFailed(RuntimeError):
    """A backend failure raised from inside an enlisted operation."""


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
    connection.rollback()

    assert connection.visible_objects == {}
    assert connection.visible_bindings == {}


def test_commit_makes_records_and_reference_visible_together() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    attempt = commit_one_attempt(store)
    binding_key = output_reference_binding_key(attempt.identity)
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)

    assert connection.visible_objects == {}
    assert connection.visible_bindings == {}

    connection.commit()

    assert member_key in connection.visible_objects
    assert binding_key in connection.visible_bindings
    assert (
        connection.visible_bindings[binding_key].schema
        == ATTEMPT_RECORD_OBJECT_SCHEMA
    )
    assert connection.staged_objects == {}
    assert connection.staged_bindings == {}


def test_entry_point_never_ends_the_callers_transaction() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    commit_one_attempt(store)

    assert connection.commit_calls == 0
    assert connection.rollback_calls == 0
    assert connection.close_calls == 0


def test_every_evidence_write_goes_through_the_passed_connection() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    commit_one_attempt(store)

    assert store.bypass_writes == 0
    assert connection.staged_objects
    assert connection.staged_bindings


def test_a_stale_first_writer_wins_member_winner_is_not_accepted() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)
    attempt = attempt_record()
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)
    stale = ObjectReference.for_record(
        SAMPLE_RECORD_OBJECT_SCHEMA, {"written": "by-another-attempt"}
    )
    connection.visible_bindings[member_key] = stale

    with pytest.raises(BindingConflictError) as raised:
        commit_evaluation_evidence(
            connection,  # ty: ignore[invalid-argument-type]
            object_store=store,
            attempt=attempt,
            samples=(sample_record(),),
        )

    assert raised.value.key == member_key
    assert raised.value.existing == stale
    assert store.calls == ["put_many_enlisted"]


def test_rebinding_a_changed_attempt_raises_the_binding_conflict() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)
    attempt = attempt_record()
    binding_key = output_reference_binding_key(attempt.identity)
    occupied = ObjectReference.for_record(
        ATTEMPT_RECORD_OBJECT_SCHEMA, {"attempt": "an-earlier-body"}
    )
    connection.visible_bindings[binding_key] = occupied

    with pytest.raises(BindingConflictError) as raised:
        commit_evaluation_evidence(
            connection,  # ty: ignore[invalid-argument-type]
            object_store=store,
            attempt=attempt,
            samples=(sample_record(),),
        )

    assert raised.value.key == binding_key
    assert raised.value.existing == occupied


@pytest.mark.parametrize("failing_op", ("put_enlisted", "bind_enlisted"))
def test_an_enlisted_failure_propagates_and_leaves_rollback_to_the_caller(
    failing_op: str,
) -> None:
    connection = FakeConnection()
    store = RaisingEnlistedObjectStore(
        connection=connection, fail_on=failing_op
    )

    with pytest.raises(_EnlistedWriteFailed, match=failing_op):
        commit_one_attempt(store)

    assert store.calls[-1] == failing_op
    assert connection.rollback_calls == 0
    assert connection.commit_calls == 0
    assert connection.close_calls == 0
    assert connection.staged_objects
    assert connection.visible_objects == {}


def test_binding_reference_resolves_the_attempt_record() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)

    attempt = commit_one_attempt(store)
    connection.commit()

    bound = connection.visible_bindings[
        output_reference_binding_key(attempt.identity)
    ]
    stored = connection.visible_objects[bound.content_hash]
    assert stored.schema == ATTEMPT_RECORD_OBJECT_SCHEMA
    assert stored.record == attempt.model_dump(mode="json")


def test_member_records_bind_under_the_attempt_prefix() -> None:
    attempt = EvaluationAttemptIdentity(attempt_id=_ATTEMPT_ID)
    binding_key = output_reference_binding_key(attempt)
    member_key = sample_record_binding_key(attempt, ordinal=3)

    assert binding_key == f"{OUTPUT_REFERENCE_BINDING_PREFIX}{_ATTEMPT_ID}"
    assert member_key == f"{binding_key}/3"


def test_evidence_wire_literals() -> None:
    assert ATTEMPT_RECORD_OBJECT_SCHEMA == (
        "dr-code/evaluation-attempt-record-v1"
    )
    assert OUTPUT_REFERENCE_BINDING_PREFIX == "dr-code/evaluation-attempt/"
    attempt = EvaluationAttemptIdentity(attempt_id=_ATTEMPT_ID)
    assert output_reference_binding_key(attempt) == (
        "dr-code/evaluation-attempt/00000000-0000-0000-0000-000000000002"
    )
    assert sample_record_binding_key(attempt, ordinal=3) == (
        "dr-code/evaluation-attempt/00000000-0000-0000-0000-000000000002/3"
    )


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


def test_evidence_rejects_misaligned_sample_records() -> None:
    connection = FakeConnection()
    store = FakeEnlistedObjectStore(connection=connection)
    attempt = attempt_record(
        members=(
            EvaluationMemberRecord(
                slot=sample_record().slot,
                sample=sample_identity(sample_id="other-sample"),
                record=reference(),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="sample record identity does not match ordered membership",
    ):
        commit_evaluation_evidence(
            store.connection,  # ty: ignore[invalid-argument-type]
            object_store=store,
            attempt=attempt,
            samples=(sample_record(),),
        )
    assert store.calls == []
