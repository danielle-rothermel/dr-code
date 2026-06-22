"""Mongo-backed eval run metadata owned by dr-code."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from dr_code.models.attempts import AttemptRecord
from dr_code.models.base import FrozenModel
from dr_code.pipeline.mongo import _database_name, mongodb_url

EVAL_RUN_METADATA_COLLECTION = "eval_run_metadata"


class EvalSeedSource(StrEnum):
    """Source used to seed an eval run."""

    ATTEMPTS_PATH = "attempts_path"
    DUMP_DIR = "dump_dir"
    IN_MEMORY = "in_memory"


class EvalRunAlreadySeededError(RuntimeError):
    """Raised when an eval run already has seed metadata."""


class EvalRunInitMetadata(FrozenModel):
    """Eval-specific metadata captured when a run is initialized."""

    worker_spec: str
    workers_by_stage: dict[str, int]
    initialized_at: str


class EvalRunSeedMetadata(FrozenModel):
    """Eval-specific metadata captured when a run is seeded."""

    source: EvalSeedSource
    source_path: str | None
    task_indices: tuple[int, ...]
    limit_per_task: int | None
    expected_jobs: int
    sample_ids_hash: str
    task_ids: tuple[str, ...]
    seeded_at: str


class EvalRunMetadata(FrozenModel):
    """Persisted eval-specific run metadata."""

    run_id: str
    init: EvalRunInitMetadata
    seed: EvalRunSeedMetadata | None = None


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def build_init_metadata(
    *,
    worker_spec: str,
    workers_by_stage: dict[str, int],
) -> EvalRunInitMetadata:
    return EvalRunInitMetadata(
        worker_spec=worker_spec,
        workers_by_stage=workers_by_stage,
        initialized_at=utc_now_iso(),
    )


def build_seed_metadata(
    *,
    records: list[AttemptRecord],
    source: EvalSeedSource,
    source_path: Path | str | None,
    task_indices: list[int] | tuple[int, ...] | None,
    limit_per_task: int | None,
) -> EvalRunSeedMetadata:
    return EvalRunSeedMetadata(
        source=source,
        source_path=_normalize_source_path(source_path),
        task_indices=tuple(task_indices or ()),
        limit_per_task=limit_per_task,
        expected_jobs=len(records),
        sample_ids_hash=_hash_sample_ids(records),
        task_ids=tuple(sorted({record.task_id for record in records})),
        seeded_at=utc_now_iso(),
    )


class EvalRunMetadataStore:
    """Persistence adapter for eval-specific run metadata."""

    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str = EVAL_RUN_METADATA_COLLECTION,
        client: MongoClient | None = None,
    ) -> None:
        resolved_url = url or mongodb_url()
        self._owns_client = client is None
        self._client = client or MongoClient(resolved_url)
        database = self._client.get_database(_database_name(resolved_url))
        self._collection: Collection = database[collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self._collection.create_index([("run_id", ASCENDING)], unique=True)

    def record_init(
        self,
        *,
        run_id: str,
        metadata: EvalRunInitMetadata,
        overwrite: bool = False,
    ) -> EvalRunMetadata:
        document = {
            "run_id": run_id,
            "init": metadata.model_dump(mode="json"),
            "seed": None,
        }
        if overwrite:
            self._collection.replace_one(
                {"run_id": run_id},
                document,
                upsert=True,
            )
        else:
            try:
                self._collection.insert_one(document)
            except DuplicateKeyError:
                existing = self.get(run_id)
                if existing is None:
                    raise
                return existing
        return EvalRunMetadata.model_validate(document)

    def record_seed(
        self,
        *,
        run_id: str,
        metadata: EvalRunSeedMetadata,
    ) -> EvalRunMetadata:
        result = self._collection.update_one(
            {"run_id": run_id, "seed": None},
            {"$set": {"seed": metadata.model_dump(mode="json")}},
        )
        if result.matched_count == 0:
            existing = self.get(run_id)
            if existing is not None and existing.seed is not None:
                msg = f"Eval run {run_id!r} already has seed metadata."
                raise EvalRunAlreadySeededError(msg)
            msg = f"Eval run {run_id!r} does not have init metadata."
            raise RuntimeError(msg)
        stored = self.get(run_id)
        if stored is None:
            msg = f"Eval run metadata for {run_id!r} was not persisted."
            raise RuntimeError(msg)
        return stored

    def get(self, run_id: str) -> EvalRunMetadata | None:
        document = self._collection.find_one({"run_id": run_id}, {"_id": 0})
        if document is None:
            return None
        return EvalRunMetadata.model_validate(document)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def _hash_sample_ids(records: list[AttemptRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalize_source_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())
