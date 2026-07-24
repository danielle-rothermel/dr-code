from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import duckdb
import pytest
from fastapi.testclient import TestClient

from dr_code.viewer.analytics import ViewerAnalytics
from dr_code.viewer.app import create_app
from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import (
    InvalidTaskAnnotationError,
    MachineTaskAnnotationWriteOutcome,
    TaskAnnotationOrigin,
    TaskAnnotationProvenance,
    TaskNotFoundError,
    TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
    decode_task_annotation_provenance,
    encode_task_annotation_provenance,
    validate_task_identity,
)
from viewer.helpers import write_bundle

REAL_DATASET = "evalplus/humanevalplus"
REAL_TASK = "HumanEval/5"


def _task_identity(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()


REAL_TASK_IDENTITY = _task_identity(REAL_TASK)


@pytest.fixture
def static_viewer_dir(tmp_path: Path) -> Path:
    frontend = tmp_path / "static"
    frontend.mkdir()
    (frontend / "index.html").write_text("<main>viewer fixture</main>")
    return frontend


def _provenance() -> TaskAnnotationProvenance:
    return TaskAnnotationProvenance(
        model="review-model",
        taxonomy_version="taxonomy-v2",
        repeats=5,
        agreement=0.8,
        extra={
            "labels": ["algorithm", "edge-case"],
            "settings": {"temperature": 0, "enabled": True},
        },
    )


@pytest.mark.parametrize(
    "task_identity",
    ["a" * 63, "A" * 64, "g" * 64],
)
def test_task_identity_requires_lowercase_sha256(task_identity: str) -> None:
    with pytest.raises(
        InvalidTaskAnnotationError,
        match="lowercase SHA-256",
    ):
        validate_task_identity("dataset", "Task/1", task_identity)


def test_provenance_is_strict_canonical_and_defensively_immutable() -> None:
    source = {
        "labels": ["algorithm"],
        "settings": {"enabled": True},
    }
    provenance = TaskAnnotationProvenance(
        model="model",
        taxonomy_version="v1",
        repeats=3,
        agreement=2 / 3,
        extra=source,
    )
    source["labels"].append("mutated")
    source["settings"]["enabled"] = False

    encoded = encode_task_annotation_provenance(provenance)
    assert encoded == (
        '{"agreement":0.6666666666666666,'
        '"extra":{"labels":["algorithm"],"settings":{"enabled":true}},'
        '"model":"model","repeats":3,"taxonomy_version":"v1"}'
    )
    assert (
        encode_task_annotation_provenance(
            decode_task_annotation_provenance(encoded)
        )
        == encoded
    )
    with pytest.raises(TypeError):
        provenance.extra["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-json", "not valid JSON"),
        ("[]", "JSON object"),
        (
            '{"model":"m","taxonomy_version":"v","repeats":true,'
            '"agreement":1,"extra":{}}',
            "repeats must be an integer",
        ),
        (
            '{"model":"m","taxonomy_version":"v","repeats":1,'
            '"agreement":1,"extra":{},"unexpected":1}',
            "exactly",
        ),
        (
            '{"model":"m","taxonomy_version":"v","repeats":1,'
            '"agreement":NaN,"extra":{}}',
            "invalid JSON number",
        ),
        (
            '{"model":"m","model":"duplicate","taxonomy_version":"v",'
            '"repeats":1,"agreement":1,"extra":{}}',
            "keys must be unique",
        ),
    ],
)
def test_provenance_rejects_corrupt_json(payload: str, message: str) -> None:
    with pytest.raises(InvalidTaskAnnotationError, match=message):
        decode_task_annotation_provenance(payload)


@pytest.mark.parametrize("agreement", [10**1000, float("inf")])
def test_provenance_translates_invalid_agreement_numbers(
    agreement: int | float,
) -> None:
    with pytest.raises(InvalidTaskAnnotationError, match="numeric|finite"):
        TaskAnnotationProvenance(agreement=agreement)


@pytest.mark.parametrize(
    "extra",
    [
        {"just_above": TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX + 1},
        {"just_below": -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX - 1},
        {"nested": [{"huge": 10**10_000}]},
        {"nested": [{"huge": -(10**10_000)}]},
    ],
)
def test_provenance_rejects_constructed_unsafe_extra_integers(
    extra: dict[str, object],
) -> None:
    with pytest.raises(InvalidTaskAnnotationError, match="safe integer"):
        TaskAnnotationProvenance(extra=extra)


def test_provenance_preserves_safe_extra_integer_boundaries_and_booleans() -> (
    None
):
    provenance = TaskAnnotationProvenance(
        extra={
            "minimum": -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
            "maximum": TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
            "nested": [
                True,
                {"minimum": -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX},
            ],
        }
    )

    encoded = encode_task_annotation_provenance(provenance)

    assert json.loads(encoded)["extra"] == {
        "minimum": -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
        "maximum": TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
        "nested": [
            True,
            {"minimum": -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX},
        ],
    }
    assert decode_task_annotation_provenance(encoded) == provenance


@pytest.mark.parametrize(
    "extra",
    [
        {"nested": [TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX + 1]},
        {"nested": [-TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX - 1]},
    ],
)
def test_provenance_rejects_unsafe_extra_integers_during_json_decode(
    extra: dict[str, object],
) -> None:
    payload = json.dumps(
        {
            "agreement": None,
            "extra": extra,
            "model": None,
            "repeats": None,
            "taxonomy_version": None,
        }
    )

    with pytest.raises(InvalidTaskAnnotationError, match="safe integer"):
        decode_task_annotation_provenance(payload)


def test_provenance_rejects_huge_extra_integer_during_json_decode() -> None:
    huge_integer = "1" + ("0" * 5_000)
    payload = (
        '{"agreement":null,"extra":{"huge":'
        + huge_integer
        + '},"model":null,"repeats":null,"taxonomy_version":null}'
    )

    with pytest.raises(InvalidTaskAnnotationError, match="4300 digits"):
        decode_task_annotation_provenance(payload)


@pytest.mark.parametrize("kind", ["cyclic", "deep"])
def test_provenance_translates_recursive_constructed_extra(kind: str) -> None:
    if kind == "cyclic":
        extra: dict[str, object] = {}
        extra["self"] = extra
    else:
        nested: list[object] = []
        extra = {"nested": nested}
        for _ in range(2_000):
            child: list[object] = []
            nested.append(child)
            nested = child

    with pytest.raises(
        InvalidTaskAnnotationError, match="cyclic or too deeply nested"
    ):
        TaskAnnotationProvenance(extra=extra)


def test_task_annotations_round_trip_export_and_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database:
        zeta = database.create_tag("zeta")
        alpha = database.create_tag("Alpha")
        result = database.put_machine_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
            category="reasoning",
            note="aggregate result",
            tag_ids=[zeta.tag_id, alpha.tag_id],
            provenance=_provenance(),
        )
        assert result.outcome is MachineTaskAnnotationWriteOutcome.WRITTEN
        first_export = database.export_task_annotations()
        assert first_export == database.export_task_annotations()

    with ViewerDatabase(path) as reopened:
        annotation = reopened.get_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
        )
        second_export = reopened.export_task_annotations()

    assert annotation is not None
    assert annotation.origin is TaskAnnotationOrigin.MACHINE
    assert annotation.provenance == _provenance()
    assert [tag.name for tag in annotation.tags] == ["Alpha", "zeta"]
    assert second_export == first_export
    assert second_export == [
        {
            "identity": {
                "dataset_id": "Task",
                "task_id": "Task/2",
                "task_identity": _task_identity("Task/2"),
            },
            "origin": "machine",
            "category": "reasoning",
            "note": "aggregate result",
            "tags": ["Alpha", "zeta"],
            "provenance": {
                "model": "review-model",
                "taxonomy_version": "taxonomy-v2",
                "repeats": 5,
                "agreement": 0.8,
                "extra": {
                    "labels": ["algorithm", "edge-case"],
                    "settings": {
                        "enabled": True,
                        "temperature": 0,
                    },
                },
            },
        }
    ]
    serialized = json.dumps(second_export)
    assert "created_at" not in serialized
    assert "updated_at" not in serialized
    assert "corpus_sha256" not in serialized
    assert "run_id" not in serialized


def test_task_annotation_export_uses_one_consistent_query_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database, ViewerDatabase(path) as writer:
        database.put_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
            category="before-one",
        )
        database.put_task_annotation(
            "dataset",
            "Task/2",
            _task_identity("Task/2"),
            category="before-two",
        )
        before = database.export_task_annotations()

        connection = database.connection

        class InterleavedConnection:
            fired = False

            def execute(
                self,
                query: str,
                parameters: object = None,
            ) -> duckdb.DuckDBPyConnection:
                if parameters is None:
                    result = connection.execute(query)
                else:
                    result = connection.execute(query, parameters)
                if (
                    not self.fired
                    and "FROM task_annotations AS a" in query
                    and "AS tag_names" in query
                ):
                    self.fired = True
                    writer.connection.execute("BEGIN TRANSACTION")
                    writer.connection.execute(
                        """
                        UPDATE task_annotations
                        SET category = 'after-one'
                        WHERE dataset_id = 'dataset' AND task_id = 'Task/1'
                        """
                    )
                    writer.connection.execute(
                        """
                        DELETE FROM task_annotations
                        WHERE dataset_id = 'dataset' AND task_id = 'Task/2'
                        """
                    )
                    writer.connection.execute("COMMIT")
                return result

        proxy = InterleavedConnection()
        database._connection = cast(  # noqa: SLF001
            duckdb.DuckDBPyConnection, proxy
        )
        try:
            interleaved = database.export_task_annotations()
        finally:
            database._connection = connection  # noqa: SLF001
        after = writer.export_task_annotations()

    assert proxy.fired
    assert interleaved in (before, after)
    assert before != after


def test_task_annotation_get_uses_one_consistent_query_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database, ViewerDatabase(path) as writer:
        before_tag = database.create_tag("before-tag")
        after_tag = database.create_tag("after-tag")
        before = database.put_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
            category="before",
            tag_ids=[before_tag.tag_id],
        )
        connection = database.connection

        class InterleavedConnection:
            fired = False

            def execute(
                self,
                query: str,
                parameters: object = None,
            ) -> duckdb.DuckDBPyConnection:
                if parameters is None:
                    result = connection.execute(query)
                else:
                    result = connection.execute(query, parameters)
                if (
                    not self.fired
                    and "FROM task_annotations" in query
                    and "WHERE" in query
                ):
                    self.fired = True
                    writer.put_task_annotation(
                        "dataset",
                        "Task/1",
                        _task_identity("Task/1"),
                        category="after",
                        tag_ids=[after_tag.tag_id],
                    )
                return result

        proxy = InterleavedConnection()
        database._connection = cast(  # noqa: SLF001
            duckdb.DuckDBPyConnection, proxy
        )
        try:
            interleaved = database.get_task_annotation(
                "dataset",
                "Task/1",
                _task_identity("Task/1"),
            )
        finally:
            database._connection = connection  # noqa: SLF001
        after = writer.get_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
        )

    assert proxy.fired
    assert interleaved in (before, after)
    assert before != after


def test_human_write_replaces_machine_and_machine_cannot_replace_human() -> (
    None
):
    with ViewerDatabase(":memory:") as database:
        first = database.put_machine_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
            category="machine",
            provenance=_provenance(),
        )
        assert first.outcome is MachineTaskAnnotationWriteOutcome.WRITTEN

        human = database.put_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
            category="human",
            note="reviewed",
        )
        protected = database.put_machine_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
            category="new-machine",
            provenance=_provenance(),
        )

    assert human.origin is TaskAnnotationOrigin.HUMAN
    assert human.provenance is None
    assert protected.outcome is MachineTaskAnnotationWriteOutcome.PROTECTED
    assert protected.annotation == human


def test_interleaved_human_commit_is_atomically_protected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as initial:
        machine_tag = initial.create_tag("machine-selected")
    with ViewerDatabase(path) as human_database:

        class InterleavedMachineDatabase(ViewerDatabase):
            inserted_human = False

            def _validate_known_tags(self, tag_ids: tuple[str, ...]) -> None:
                super()._validate_known_tags(tag_ids)
                if self.inserted_human:
                    return
                self.inserted_human = True
                human_database.put_task_annotation(
                    "Task",
                    "Task/2",
                    _task_identity("Task/2"),
                    category="human won",
                )

        with InterleavedMachineDatabase(path) as machine_database:
            result = machine_database.put_machine_task_annotation(
                "Task",
                "Task/2",
                _task_identity("Task/2"),
                category="machine",
                tag_ids=[machine_tag.tag_id],
                provenance=_provenance(),
            )

    assert result.outcome is MachineTaskAnnotationWriteOutcome.PROTECTED
    assert result.annotation.origin is TaskAnnotationOrigin.HUMAN
    assert result.annotation.category == "human won"
    assert result.annotation.tags == ()


def test_human_write_retries_concurrent_machine_commit_and_wins(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    with (
        ViewerDatabase(path) as human_database,
        ViewerDatabase(path) as machine_database,
    ):
        connection = human_database.connection

        class InterleavedConnection:
            fired = False

            def execute(
                self,
                query: str,
                parameters: object = None,
            ) -> duckdb.DuckDBPyConnection:
                if query == "COMMIT" and not self.fired:
                    self.fired = True
                    machine_database.put_machine_task_annotation(
                        "Task",
                        "Task/2",
                        _task_identity("Task/2"),
                        category="concurrent machine",
                        provenance=_provenance(),
                    )
                if parameters is None:
                    return connection.execute(query)
                return connection.execute(query, parameters)

        proxy = InterleavedConnection()
        human_database._connection = cast(  # noqa: SLF001
            duckdb.DuckDBPyConnection, proxy
        )
        try:
            human = human_database.put_task_annotation(
                "Task",
                "Task/2",
                _task_identity("Task/2"),
                category="human wins",
            )
        finally:
            human_database._connection = connection  # noqa: SLF001
        stored = machine_database.get_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
        )

    assert proxy.fired
    assert human.origin is TaskAnnotationOrigin.HUMAN
    assert human.category == "human wins"
    assert stored == human


def test_transaction_preserves_error_when_rollback_is_already_inactive() -> (
    None
):
    with ViewerDatabase(":memory:") as database:
        connection = database.connection

        class InactiveRollbackConnection:
            def execute(
                self,
                query: str,
                parameters: object = None,
            ) -> duckdb.DuckDBPyConnection:
                if query == "ROLLBACK":
                    raise duckdb.TransactionException(
                        "TransactionContext Error: cannot rollback - "
                        "no transaction is active"
                    )
                if parameters is None:
                    return connection.execute(query)
                return connection.execute(query, parameters)

        database._connection = cast(  # noqa: SLF001
            duckdb.DuckDBPyConnection, InactiveRollbackConnection()
        )

        def fail() -> None:
            raise RuntimeError("original failure")

        try:
            with pytest.raises(RuntimeError, match="original failure"):
                database._transaction(fail)  # noqa: SLF001
        finally:
            database._connection = connection  # noqa: SLF001


def test_task_tags_are_atomic_and_delete_removes_links() -> None:
    with ViewerDatabase(":memory:") as database:
        tag = database.create_tag("algorithm")
        original = database.put_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
            category="original",
            tag_ids=[tag.tag_id],
        )
        with pytest.raises(InvalidTaskAnnotationError, match="unknown tag"):
            database.put_task_annotation(
                "Task",
                "Task/2",
                _task_identity("Task/2"),
                category="must roll back",
                tag_ids=["missing"],
            )
        assert (
            database.get_task_annotation(
                "Task",
                "Task/2",
                _task_identity("Task/2"),
            )
            == original
        )
        assert database.delete_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
        )
        assert not database.delete_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
        )
        assert database.connection.execute(
            "SELECT count(*) FROM task_annotation_tags"
        ).fetchone() == (0,)


def test_corrupt_persisted_provenance_is_rejected() -> None:
    with ViewerDatabase(":memory:") as database:
        database.put_machine_task_annotation(
            "Task",
            "Task/2",
            _task_identity("Task/2"),
            category="machine",
            provenance=_provenance(),
        )
        database.connection.execute(
            """
            UPDATE task_annotations
            SET provenance = '{"model":"missing exact shape"}'
            """
        )
        with pytest.raises(InvalidTaskAnnotationError, match="exactly"):
            database.get_task_annotation(
                "Task",
                "Task/2",
                _task_identity("Task/2"),
            )


def test_service_requires_registered_task_and_is_run_independent(
    tmp_path: Path,
) -> None:
    baseline = write_bundle(
        tmp_path / "baseline",
        run_id="baseline",
        dataset_id=REAL_DATASET,
        task_namespace="HumanEval",
    )
    candidate = write_bundle(
        tmp_path / "candidate",
        run_id="candidate",
        corpus_path=baseline.corpus_path,
        dataset_id=REAL_DATASET,
        task_namespace="HumanEval",
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [baseline, candidate])
        assert (
            analytics.example(baseline.run_id, "pass").task_identity
            == REAL_TASK_IDENTITY
        )
        assert (
            analytics.example(
                baseline.run_id,
                "no-code",
            ).task_identity
            is None
        )
        stored = analytics.put_task_annotation(
            REAL_DATASET,
            REAL_TASK,
            REAL_TASK_IDENTITY,
            category="shared",
        )
        assert (
            analytics.get_task_annotation(
                REAL_DATASET,
                REAL_TASK,
                REAL_TASK_IDENTITY,
            )
            == stored
        )
        with pytest.raises(TaskNotFoundError, match="registered"):
            analytics.put_task_annotation(
                REAL_DATASET,
                "HumanEval/999",
                _task_identity("HumanEval/999"),
            )
        with pytest.raises(TaskNotFoundError, match="registered"):
            analytics.put_task_annotation(
                REAL_DATASET,
                "HumanEval/0",
                _task_identity("HumanEval/0"),
            )

        exported = analytics.export_task_annotations()

    assert exported[0]["identity"] == {
        "dataset_id": REAL_DATASET,
        "task_id": REAL_TASK,
        "task_identity": REAL_TASK_IDENTITY,
    }
    assert "corpus_sha256" not in exported[0]


def test_same_task_id_in_two_registered_datasets_does_not_collide(
    tmp_path: Path,
) -> None:
    first = write_bundle(
        tmp_path / "first",
        run_id="first",
        dataset_id="dataset/one",
        task_namespace="SharedTask",
    )
    second = write_bundle(
        tmp_path / "second",
        run_id="second",
        dataset_id="dataset/two",
        corpus_path=first.corpus_path,
        task_namespace="SharedTask",
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [first, second])
        first_annotation = analytics.put_task_annotation(
            "dataset/one",
            "SharedTask/5",
            _task_identity("SharedTask/5"),
            category="first",
        )
        second_annotation = analytics.put_task_annotation(
            "dataset/two",
            "SharedTask/5",
            _task_identity("SharedTask/5"),
            category="second",
        )

    assert first_annotation.identity != second_annotation.identity
    assert first_annotation.category == "first"
    assert second_annotation.category == "second"


def test_changed_task_content_does_not_reuse_annotation_or_tag_links() -> None:
    first_identity = "1" * 64
    second_identity = "2" * 64
    with ViewerDatabase(":memory:") as database:
        tag = database.create_tag("first content")
        first = database.put_task_annotation(
            "dataset",
            "Task/5",
            first_identity,
            category="first",
            tag_ids=[tag.tag_id],
        )
        second = database.put_task_annotation(
            "dataset",
            "Task/5",
            second_identity,
            category="second",
        )

        assert (
            database.get_task_annotation(
                "dataset",
                "Task/5",
                first_identity,
            )
            == first
        )
        assert (
            database.get_task_annotation(
                "dataset",
                "Task/5",
                second_identity,
            )
            == second
        )

    assert first.identity != second.identity
    assert [item.name for item in first.tags] == ["first content"]
    assert second.tags == ()


def test_reregister_replaces_only_supplied_run_task_membership(
    tmp_path: Path,
) -> None:
    old = write_bundle(
        tmp_path / "old",
        run_id="old",
        dataset_id="dataset/old",
        task_namespace="OldTask",
    )
    current = write_bundle(
        tmp_path / "current",
        run_id="current",
        dataset_id="dataset/current",
        task_namespace="CurrentTask",
    )
    old_without_evaluation = write_bundle(
        tmp_path / "old-without-evaluation",
        run_id="old",
        dataset_id="dataset/old",
        task_namespace="OldTask",
        with_evaluation=False,
    )
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database:
        analytics = ViewerAnalytics(database, [old])
        old_annotation = analytics.put_task_annotation(
            "dataset/old",
            "OldTask/5",
            _task_identity("OldTask/5"),
            category="keep",
        )

    with ViewerDatabase(path) as database:
        analytics = ViewerAnalytics(database, [current])
        assert database.task_is_registered(old_annotation.identity)
        current_annotation = analytics.put_task_annotation(
            "dataset/current",
            "CurrentTask/5",
            _task_identity("CurrentTask/5"),
            category="active",
        )
        assert (
            database.get_task_annotation(
                "dataset/old",
                "OldTask/5",
                _task_identity("OldTask/5"),
            )
            == old_annotation
        )

        database.register_runs(())

        assert database.task_is_registered(old_annotation.identity)
        assert database.task_is_registered(current_annotation.identity)

        database.register_runs([old_without_evaluation])

        assert not database.task_is_registered(old_annotation.identity)
        assert database.task_is_registered(current_annotation.identity)
        assert (
            database.get_task_annotation(
                "dataset/old",
                "OldTask/5",
                _task_identity("OldTask/5"),
            )
            == old_annotation
        )
        assert (
            database.get_task_annotation(
                "dataset/current",
                "CurrentTask/5",
                _task_identity("CurrentTask/5"),
            )
            == current_annotation
        )


def test_preprocessing_only_run_cannot_authorize_task_annotations(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id=REAL_DATASET,
        task_namespace="HumanEval",
        with_evaluation=False,
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])

        assert (
            analytics.example(descriptor.run_id, "no-code").dataset_id is None
        )
        assert not database.task_is_registered(
            validate_task_identity(
                REAL_DATASET,
                REAL_TASK,
                REAL_TASK_IDENTITY,
            )
        )
        with pytest.raises(TaskNotFoundError, match="registered"):
            analytics.put_task_annotation(
                REAL_DATASET,
                REAL_TASK,
                REAL_TASK_IDENTITY,
                category="unauthenticated",
            )
        with pytest.raises(TaskNotFoundError, match="registered"):
            analytics.put_machine_task_annotation(
                REAL_DATASET,
                REAL_TASK,
                REAL_TASK_IDENTITY,
                category="unauthenticated",
                provenance=_provenance(),
            )


def test_http_task_contract_forces_human_and_uses_404_422_semantics(
    tmp_path: Path,
    static_viewer_dir: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id=REAL_DATASET,
        task_namespace="HumanEval",
    )
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        analytics.put_machine_task_annotation(
            REAL_DATASET,
            REAL_TASK,
            REAL_TASK_IDENTITY,
            category="machine",
            provenance=_provenance(),
        )
        client = TestClient(
            create_app(analytics, static_dir=static_viewer_dir),
            base_url="http://127.0.0.1",
        )
        params = {
            "dataset_id": REAL_DATASET,
            "task_id": REAL_TASK,
            "task_identity": REAL_TASK_IDENTITY,
        }

        machine = client.get("/api/task-annotations", params=params)
        injected = client.put(
            "/api/task-annotations",
            params=params,
            json={
                "category": "attempt",
                "note": None,
                "tag_ids": [],
                "origin": "machine",
                "provenance": {
                    "model": "injected",
                    "taxonomy_version": None,
                    "repeats": 1,
                    "agreement": 1,
                    "extra": {},
                },
            },
        )
        human = client.put(
            "/api/task-annotations",
            params=params,
            json={
                "category": " human ",
                "note": "reviewed",
                "tag_ids": [],
            },
        )
        exported = client.get("/api/task-annotations/export")
        missing_task = client.get(
            "/api/task-annotations",
            params={
                "dataset_id": REAL_DATASET,
                "task_id": "HumanEval/999",
                "task_identity": _task_identity("HumanEval/999"),
            },
        )
        unknown_target = client.delete(
            "/api/task-annotations",
            params={
                "dataset_id": REAL_DATASET,
                "task_id": "HumanEval/999",
                "task_identity": _task_identity("HumanEval/999"),
            },
        )
        deleted = client.delete("/api/task-annotations", params=params)
        missing_annotation = client.delete(
            "/api/task-annotations", params=params
        )

    assert machine.status_code == 200
    assert machine.json()["origin"] == "machine"
    assert machine.json()["provenance"]["extra"]["labels"] == [
        "algorithm",
        "edge-case",
    ]
    assert injected.status_code == 422
    assert human.status_code == 200
    assert human.json()["origin"] == "human"
    assert human.json()["provenance"] is None
    assert human.json()["category"] == "human"
    assert exported.status_code == 200
    assert exported.json() == [
        {
            "identity": {
                "dataset_id": REAL_DATASET,
                "task_id": REAL_TASK,
                "task_identity": REAL_TASK_IDENTITY,
            },
            "origin": "human",
            "category": "human",
            "note": "reviewed",
            "tags": [],
            "provenance": None,
        }
    ]
    assert missing_task.status_code == 404
    assert unknown_target.status_code == 404
    assert deleted.status_code == 204
    assert missing_annotation.status_code == 204


def test_http_task_annotation_preserves_safe_provenance_integers(
    tmp_path: Path,
    static_viewer_dir: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id=REAL_DATASET,
        task_namespace="HumanEval",
    )
    extra = {
        "minimum": -TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
        "maximum": TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX,
        "nested": [True, {"maximum": TASK_PROVENANCE_JSON_SAFE_INTEGER_MAX}],
    }
    with ViewerDatabase(":memory:") as database:
        analytics = ViewerAnalytics(database, [descriptor])
        analytics.put_machine_task_annotation(
            REAL_DATASET,
            REAL_TASK,
            REAL_TASK_IDENTITY,
            category="machine",
            provenance=TaskAnnotationProvenance(extra=extra),
        )
        client = TestClient(
            create_app(analytics, static_dir=static_viewer_dir),
            base_url="http://127.0.0.1",
        )
        response = client.get(
            "/api/task-annotations",
            params={
                "dataset_id": REAL_DATASET,
                "task_id": REAL_TASK,
                "task_identity": REAL_TASK_IDENTITY,
            },
        )

    assert response.status_code == 200
    assert json.loads(response.text)["provenance"]["extra"] == extra
