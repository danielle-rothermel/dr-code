from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from dr_exec import AutoPoolCapacity, CancelledOutcome, ExecutionPoolConfig
from dr_store import ArtifactBundlePublication, MemoryBackend, ObjectStore

from _executor_stubs import importable_json_executor, scripted_executor
from dr_code.evaluation import (
    RecordPlacement,
    ShardLimits,
    evaluate_batch,
)

from ._batch_builders import BatchStore, cache, request
from ._bundle_builders import publish_batch

pytestmark = pytest.mark.asyncio


async def test_bundle_local_publication_writes_one_terminal_closed_bundle(
    tmp_path: Path,
) -> None:
    result, _ = await publish_batch(tmp_path)
    assert result.bundle_path is not None
    assert (result.bundle_path / "manifest.json").is_file()
    assert {path.name for path in result.bundle_path.iterdir()} == {
        "manifest.json",
        "evaluation-attempt.json",
        "sample-records-00000000.jsonl",
        "projection-evaluation-samples.jsonl",
        "projection-materialized-candidates.jsonl",
        "projection-metric-records.jsonl",
        "projection-aggregation-results.jsonl",
        "projection-scores.jsonl",
    }


async def test_truthful_placement_and_projection_preconditions(
    tmp_path: Path,
) -> None:
    executor = importable_json_executor()
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(ValueError, match="bundle-local"):
            await evaluate_batch(
                request(projections=()),
                executor=executor,
                execution_cache=execution_cache,
                object_store=None,
                publication=None,
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
            )
        object_request = request().model_copy(
            update={"record_placement": RecordPlacement.OBJECT_STORE}
        )
        with pytest.raises(ValueError, match="requested projections"):
            await evaluate_batch(
                object_request,
                executor=executor,
                execution_cache=execution_cache,
                object_store=ObjectStore(MemoryBackend()),
                publication=None,
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
            )
    finally:
        await execution_cache.close()


async def test_object_store_without_projections_needs_no_bundle(
    tmp_path: Path,
) -> None:
    batch_request = request(projections=()).model_copy(
        update={"record_placement": RecordPlacement.OBJECT_STORE}
    )
    execution_cache = cache(BatchStore())
    try:
        result = await evaluate_batch(
            batch_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=ObjectStore(MemoryBackend()),
            publication=None,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()
    assert result.bundle_path is None
    assert result.projections == ()
    assert result.attempt.members[0].record is not None
    assert result.attempt.members[0].record.kind == "stored_record"


async def test_shard_count_limit_splits_bundle_local_records(
    tmp_path: Path,
) -> None:
    batch_request = request(2, projections=()).model_copy(
        update={
            "shard_limits": ShardLimits(
                max_records=1,
                max_uncompressed_bytes=10_000_000,
            )
        }
    )
    publication = ArtifactBundlePublication.allocate(tmp_path, prefix="count")
    execution_cache = cache(BatchStore(), resident=1)
    try:
        result = await evaluate_batch(
            batch_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=None,
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()
    assert result.bundle_path is not None
    assert (result.bundle_path / "sample-records-00000000.jsonl").is_file()
    assert (result.bundle_path / "sample-records-00000001.jsonl").is_file()


async def test_oversized_single_record_fails_before_terminal_publication(
    tmp_path: Path,
) -> None:
    batch_request = request(projections=()).model_copy(
        update={
            "shard_limits": ShardLimits(
                max_records=1,
                max_uncompressed_bytes=1,
            )
        }
    )
    publication = ArtifactBundlePublication.allocate(tmp_path, prefix="bytes")
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(ValueError, match="exceeds shard"):
            await evaluate_batch(
                batch_request,
                executor=importable_json_executor(),
                execution_cache=execution_cache,
                object_store=None,
                publication=publication,
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
            )
    finally:
        await execution_cache.close()
    assert not (publication.path / "manifest.json").exists()


async def test_cancellation_never_terminally_publishes(
    tmp_path: Path,
) -> None:
    publication = ArtifactBundlePublication.allocate(tmp_path, prefix="cancel")
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(asyncio.CancelledError):
            await evaluate_batch(
                request(projections=()),
                executor=scripted_executor(outcome=CancelledOutcome()),
                execution_cache=execution_cache,
                object_store=None,
                publication=publication,
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
            )
    finally:
        await execution_cache.close()
    assert not (publication.path / "manifest.json").exists()
