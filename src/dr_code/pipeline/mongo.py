"""Mongo eval_results sink for test-stage outcomes."""

from __future__ import annotations

import os
from threading import Lock

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.uri_parser import parse_uri

from dr_code.models.outcomes import TestOutcome

DEFAULT_MONGODB_URL = "mongodb://localhost:27017/dr_queues"
EVAL_RESULTS_COLLECTION = "eval_results"


def mongodb_url() -> str:
    return os.environ.get("MONGODB_URL", DEFAULT_MONGODB_URL)


def _database_name(url: str) -> str:
    parsed = parse_uri(url)
    database = parsed.get("database")
    if database:
        return database
    return "dr_queues"


class EvalResultsSink:
    """Upsert TestOutcome documents keyed by (run_id, sample_id)."""

    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str = EVAL_RESULTS_COLLECTION,
        client: MongoClient | None = None,
    ) -> None:
        resolved_url = url or mongodb_url()
        self._owns_client = client is None
        self._client = client or MongoClient(resolved_url)
        database = self._client.get_database(_database_name(resolved_url))
        self._collection: Collection = database[collection_name]
        self._lock = Lock()
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._collection.create_index(
            [("run_id", ASCENDING), ("sample_id", ASCENDING)],
            unique=True,
        )
        self._collection.create_index([("run_id", ASCENDING)])
        self._collection.create_index([("task_id", ASCENDING)])
        self._collection.create_index([("outcome_kind", ASCENDING)])

    def upsert_test_outcome(
        self,
        outcome: TestOutcome,
        *,
        provenance_source: str | None = None,
        occurrence_count: int = 1,
    ) -> None:
        """Upsert one test outcome with denormalized slice fields."""
        document = outcome.model_dump(mode="json")
        document.update(
            {
                "run_id": outcome.run_id,
                "sample_id": outcome.sample_id,
                "task_id": outcome.task_id,
                "outcome_kind": outcome.outcome_kind,
                "all_tests_passed": outcome.all_tests_passed,
                "provenance_source": provenance_source,
                "occurrence_count": occurrence_count,
            },
        )
        with self._lock:
            self._collection.update_one(
                {
                    "run_id": outcome.run_id,
                    "sample_id": outcome.sample_id,
                },
                {"$set": document},
                upsert=True,
            )

    def count_by_run_id(self, run_id: str) -> int:
        return self._collection.count_documents({"run_id": run_id})

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
