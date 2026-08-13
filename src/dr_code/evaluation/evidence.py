from __future__ import annotations

from typing import Final, Protocol

from dr_serialize import Jsonable
from dr_store import (
    BindingConflictError,
    BindStatus,
    ObjectReference,
    PutStatus,
)
from sqlalchemy.engine import Connection  # noqa: TC002

from dr_code.evaluation.bundle import (
    SAMPLE_RECORD_OBJECT_SCHEMA,
)
from dr_code.evaluation.id import EvalAttemptId
from dr_code.evaluation.records import (
    EVAL_ATTEMPT_SCHEMA_VERSION,
    EvalAttemptRecord,
    SampleEvalRecord,
)
from dr_code.evaluation.references import StoredRecordReference
from dr_code.evaluation.validation import validate_sample_record_graph

ATTEMPT_RECORD_OBJECT_SCHEMA: Final = "dr-code/evaluation-attempt-record-v1"
OUTPUT_REFERENCE_BINDING_PREFIX: Final = "dr-code/evaluation-attempt/"


class EnlistedObjectStore(Protocol):
    """The dr-store enlisted surface this write path depends on.

    Enlisted operations run inside the caller's transaction: they never
    commit, roll back, or release the connection they are handed.
    """

    def put_enlisted(
        self,
        connection: Connection,
        schema: str,
        record: Jsonable,
    ) -> tuple[ObjectReference, PutStatus]: ...

    def put_many_enlisted(
        self,
        connection: Connection,
        entries: dict[str, tuple[str, Jsonable]],
    ) -> dict[str, ObjectReference]: ...

    def bind_enlisted(
        self,
        connection: Connection,
        key: str,
        reference: ObjectReference,
    ) -> BindStatus: ...


def output_reference_binding_key(
    attempt: EvalAttemptId,
    /,
) -> str:
    """Name the binding an attempt's published evidence resolves through."""

    return f"{OUTPUT_REFERENCE_BINDING_PREFIX}{attempt.attempt_id}"


def sample_record_binding_key(
    attempt: EvalAttemptId,
    /,
    *,
    ordinal: int,
) -> str:
    """Name the binding one member sample record commits under."""

    if ordinal < 0:
        raise ValueError("sample record ordinal must not be negative")
    return f"{OUTPUT_REFERENCE_BINDING_PREFIX}{attempt.attempt_id}/{ordinal}"


def commit_eval_evidence(
    connection: Connection,
    /,
    *,
    object_store: EnlistedObjectStore,
    attempt: EvalAttemptRecord,
    samples: tuple[SampleEvalRecord, ...],
) -> StoredRecordReference:
    """Write an attempt's evidence inside the caller's open transaction.

    Every member sample record, the attempt record, and the
    ``output_reference`` binding that resolves the attempt are written through
    dr-store's enlisted operations on ``connection``. They therefore become
    visible exactly when the caller commits and leave nothing behind when the
    caller rolls back. This path never commits, rolls back, or closes
    ``connection``: the transaction belongs to the caller.

    Publishing the generation artifact bundle before calling this is the
    caller's obligation; nothing here enforces the ordering.

    ``samples`` carries one record for each attempt member whose
    ``record`` reference is present, in member order. Members omitted by
    admission or retained-evidence limits carry no sample record and receive
    no binding, but their original ordinals are preserved for the members
    that do commit.

    ``put_many_enlisted`` is first-writer-wins: an occupied member key keeps
    its existing reference instead of raising. Any member key whose winner is
    not the record written here raises :class:`BindingConflictError`, so a
    collision can never commit as if it were this attempt's evidence. The
    ``output_reference`` binding raises the same error directly from dr-store,
    which is by design what re-committing a changed attempt under an existing
    ``attempt_id`` does.

    Any exception leaves the writes already issued staged in the caller's
    transaction. Rolling that transaction back is the caller's obligation:
    this path neither swallows the exception nor rolls back on the caller's
    behalf.
    """

    referenced_members = tuple(
        (ordinal, member)
        for ordinal, member in enumerate(attempt.members)
        if member.record is not None
    )
    if len(samples) != len(referenced_members):
        raise ValueError(
            "evidence sample records must match the attempt's referenced members"
        )
    for (ordinal, member), sample in zip(
        referenced_members, samples, strict=True
    ):
        validate_sample_record_graph(
            sample,
            slot=member.slot,
            sample=member.sample,
            plan=attempt.plan,
            runtime=attempt.runtime,
            cache_namespace=attempt.cache_namespace,
        )
    entries: dict[str, tuple[str, Jsonable]] = {
        sample_record_binding_key(attempt.identity, ordinal=ordinal): (
            SAMPLE_RECORD_OBJECT_SCHEMA,
            sample.model_dump(mode="json"),
        )
        for (ordinal, _member), sample in zip(
            referenced_members, samples, strict=True
        )
    }
    if entries:
        winners = object_store.put_many_enlisted(connection, entries)
        for key, (schema, record) in entries.items():
            written = ObjectReference.for_record(schema, record)
            winner = winners[key]
            if winner != written:
                raise BindingConflictError(
                    key=key,
                    existing=winner,
                    requested=written,
                )
    attempt_reference, _ = object_store.put_enlisted(
        connection,
        ATTEMPT_RECORD_OBJECT_SCHEMA,
        attempt.model_dump(mode="json"),
    )
    object_store.bind_enlisted(
        connection,
        output_reference_binding_key(attempt.identity),
        attempt_reference,
    )
    return StoredRecordReference(
        reference=attempt_reference,
        schema_version=EVAL_ATTEMPT_SCHEMA_VERSION,
    )


__all__ = [
    "ATTEMPT_RECORD_OBJECT_SCHEMA",
    "EnlistedObjectStore",
    "OUTPUT_REFERENCE_BINDING_PREFIX",
    "commit_eval_evidence",
    "output_reference_binding_key",
    "sample_record_binding_key",
]
