"""Evidence writes against a real PostgreSQL-backed dr-store.

`test_evidence.py` drives `commit_evaluation_evidence` through a fake that
mirrors dr-store's enlisted semantics, and pins the call surface with
`inspect.signature`. Matching signatures do not establish matching behavior:
whether a rollback really leaves nothing behind, and whether a first-writer-wins
collision really surfaces, are properties of the database rather than of the
call shape. These tests exercise the same entry point against a real
`ObjectStore` on a live connection.

They need a PostgreSQL server and skip without one. dr-store's scratch-server
script supplies both the server and the DSN from a consumer checkout:

    ~/drotherm/repos/dr-store/scripts/test-postgres.sh -- \
        uv run pytest -q tests/evaluation/test_evidence_postgres.py
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from dr_store import BindingConflictError, ObjectReference, ObjectStore

from _evidence_builders import attempt_record, sample_record
from dr_code.evaluation import (
    ATTEMPT_RECORD_OBJECT_SCHEMA,
    SAMPLE_RECORD_OBJECT_SCHEMA,
    commit_evaluation_evidence,
    output_reference_binding_key,
    sample_record_binding_key,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


def _dsn() -> str:
    dsn = os.environ.get("DR_STORE_POSTGRES_DSN")
    if dsn is None:
        pytest.skip("DR_STORE_POSTGRES_DSN is not configured")
    return dsn


def _sqlalchemy_dsn(dsn: str) -> str:
    # dr-store drives PostgreSQL through psycopg3; the bare `postgresql://`
    # scheme would resolve to psycopg2, which is not installed.
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn.removeprefix("postgresql://")
    return dsn


@pytest.fixture(name="engine")
def engine_fixture() -> Iterator[Engine]:
    from sqlalchemy import create_engine

    engine = create_engine(
        _sqlalchemy_dsn(_dsn()),
        connect_args={"options": "-c search_path=pg_catalog"},
    )
    try:
        yield engine
    finally:
        engine.dispose()


_DEDICATED_DATABASE = "dr_store_test"


async def _reset_schema(engine: AsyncEngine) -> None:
    """Drop dr-store's schema, refusing to run outside the test database."""
    from sqlalchemy import text

    async with engine.connect() as connection:
        database = await connection.scalar(
            text("SELECT pg_catalog.current_database()")
        )
        if database != _DEDICATED_DATABASE:
            pytest.fail(
                "PostgreSQL schema reset is allowed only in the "
                f"{_DEDICATED_DATABASE!r} database; connected to {database!r}"
            )
        await connection.execute(
            text("DROP SCHEMA IF EXISTS dr_store CASCADE")
        )
        await connection.commit()


@pytest_asyncio.fixture(name="object_store")
async def object_store_fixture() -> AsyncIterator[ObjectStore]:
    from dr_store import PostgresBackend, install_postgres
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(_sqlalchemy_dsn(_dsn()))
    try:
        # Each test owns the whole schema: evidence keys are derived from a
        # fixed attempt identity, so leftovers from a prior test would collide.
        await _reset_schema(engine)
        await install_postgres(engine)
        yield ObjectStore(await PostgresBackend.open(engine))
    finally:
        await _reset_schema(engine)
        await engine.dispose()


def _resolve(
    store: ObjectStore, connection: Connection, key: str
) -> ObjectReference | None:
    return store.resolve_enlisted(connection, key)


async def test_committed_evidence_is_visible_to_a_later_transaction(
    engine: Engine,
    object_store: ObjectStore,
) -> None:
    attempt = attempt_record()
    sample = sample_record()

    with engine.connect() as connection, connection.begin():
        stored = commit_evaluation_evidence(
            connection,
            object_store=object_store,
            attempt=attempt,
            samples=(sample,),
        )

    binding_key = output_reference_binding_key(attempt.identity)
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)
    with engine.connect() as connection, connection.begin():
        assert _resolve(object_store, connection, binding_key) == (
            stored.reference
        )
        assert object_store.get_enlisted(connection, stored.reference) == (
            attempt.model_dump(mode="json")
        )
        member = _resolve(object_store, connection, member_key)
        assert member is not None
        assert member.schema == SAMPLE_RECORD_OBJECT_SCHEMA
        assert object_store.get_enlisted(connection, member) == (
            sample.model_dump(mode="json")
        )
    assert stored.reference.schema == ATTEMPT_RECORD_OBJECT_SCHEMA


async def test_rollback_leaves_no_evidence_in_the_database(
    engine: Engine,
    object_store: ObjectStore,
) -> None:
    attempt = attempt_record()
    binding_key = output_reference_binding_key(attempt.identity)
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)

    with engine.connect() as connection:
        transaction = connection.begin()
        commit_evaluation_evidence(
            connection,
            object_store=object_store,
            attempt=attempt,
            samples=(sample_record(),),
        )
        # Visible on the caller's own connection before it decides.
        assert _resolve(object_store, connection, binding_key) is not None
        transaction.rollback()

    with engine.connect() as connection, connection.begin():
        assert _resolve(object_store, connection, binding_key) is None
        assert _resolve(object_store, connection, member_key) is None


async def test_recommitting_a_changed_attempt_raises_the_binding_conflict(
    engine: Engine,
    object_store: ObjectStore,
) -> None:
    # Re-committing under an existing attempt_id with different content is the
    # collision the write path exists to refuse. The fake asserts this; here
    # the real database is what refuses it.
    attempt = attempt_record()
    with engine.connect() as connection, connection.begin():
        commit_evaluation_evidence(
            connection,
            object_store=object_store,
            attempt=attempt,
            samples=(sample_record(),),
        )

    changed = attempt_record(cache_namespace="evaluation-v2")
    with engine.connect() as connection, connection.begin():
        with pytest.raises(BindingConflictError) as caught:
            commit_evaluation_evidence(
                connection,
                object_store=object_store,
                attempt=changed,
                samples=(sample_record(),),
            )
        assert caught.value.key == output_reference_binding_key(
            changed.identity
        )

    # The refusal left the original evidence intact and resolvable.
    with engine.connect() as connection, connection.begin():
        resolved = _resolve(
            object_store,
            connection,
            output_reference_binding_key(attempt.identity),
        )
        assert resolved is not None
        assert object_store.get_enlisted(connection, resolved) == (
            attempt.model_dump(mode="json")
        )


async def test_a_member_key_held_by_a_different_record_is_refused(
    engine: Engine,
    object_store: ObjectStore,
) -> None:
    # `put_many_enlisted` is first-writer-wins, so an occupied member key
    # returns the *existing* winner instead of raising. The write path compares
    # that winner against what it wrote and refuses the mismatch, which is what
    # stops a collision committing as if it were this attempt's evidence.
    attempt = attempt_record()
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)
    squatter = sample_record().model_dump(mode="json")
    squatter["task_id"] = "occupied-by-another-record"

    with engine.connect() as connection, connection.begin():
        object_store.put_many_enlisted(
            connection,
            {member_key: (SAMPLE_RECORD_OBJECT_SCHEMA, squatter)},
        )

    with engine.connect() as connection:
        transaction = connection.begin()
        with pytest.raises(BindingConflictError) as caught:
            commit_evaluation_evidence(
                connection,
                object_store=object_store,
                attempt=attempt,
                samples=(sample_record(),),
            )
        assert caught.value.key == member_key
        transaction.rollback()

    # The attempt binding was never reached, so nothing resolves for it.
    with engine.connect() as connection, connection.begin():
        assert (
            _resolve(
                object_store,
                connection,
                output_reference_binding_key(attempt.identity),
            )
            is None
        )


async def test_recommitting_identical_evidence_is_idempotent(
    engine: Engine,
    object_store: ObjectStore,
) -> None:
    attempt = attempt_record()
    sample = sample_record()

    with engine.connect() as connection, connection.begin():
        first = commit_evaluation_evidence(
            connection,
            object_store=object_store,
            attempt=attempt,
            samples=(sample,),
        )
    with engine.connect() as connection, connection.begin():
        second = commit_evaluation_evidence(
            connection,
            object_store=object_store,
            attempt=attempt,
            samples=(sample,),
        )
    assert first == second


async def test_a_failure_leaves_staged_writes_for_the_caller_to_roll_back(
    engine: Engine,
    object_store: ObjectStore,
) -> None:
    # The entry point never rolls back on the caller's behalf, so a mismatched
    # sample count raised after member writes leaves them staged and visible
    # on the caller's connection until the caller decides.
    attempt = attempt_record()
    member_key = sample_record_binding_key(attempt.identity, ordinal=0)

    with engine.connect() as connection:
        transaction = connection.begin()
        object_store.put_many_enlisted(
            connection,
            {
                member_key: (
                    SAMPLE_RECORD_OBJECT_SCHEMA,
                    sample_record().model_dump(mode="json"),
                )
            },
        )
        with pytest.raises(ValueError, match="match the attempt's members"):
            commit_evaluation_evidence(
                connection,
                object_store=object_store,
                attempt=attempt,
                samples=(),
            )
        assert _resolve(object_store, connection, member_key) is not None
        transaction.rollback()

    with engine.connect() as connection, connection.begin():
        assert _resolve(object_store, connection, member_key) is None
