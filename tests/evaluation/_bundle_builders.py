from __future__ import annotations

from pathlib import Path

from dr_exec import AutoPoolCapacity, ExecutionPoolConfig
from dr_store import (
    ArtifactBundlePublication,
    BundleReadLimits,
    MemoryBackend,
    ObjectStore,
)

from _executor_stubs import importable_json_executor
from dr_code.evaluation import (
    EvalBatchRequest,
    EvalReadLimits,
    ProjectionKind,
    RecordPlacement,
    StoredRecordReference,
    evaluate_batch,
)

from ._batch_builders import BatchStore, cache, request


def read_limits() -> EvalReadLimits:
    return EvalReadLimits(
        bundle=BundleReadLimits(
            manifest_max_bytes=1_000_000,
            manifest_max_depth=100,
            max_artifacts=100,
            max_bytes_per_artifact=20_000_000,
            max_total_artifact_bytes=100_000_000,
        ),
        max_sample_records=1_000,
        max_object_reads=1_000,
        max_reference_depth=10,
    )


async def stored_source_request(
    batch_request: EvalBatchRequest,
    *,
    object_store: ObjectStore,
) -> EvalBatchRequest:
    """Store each input's source object so restoring can resolve it."""

    inputs = []
    for item in batch_request.inputs:
        source_object, _ = await object_store.put(
            "tests/input",
            {"sample_id": item.sample.metadata.identity.sample_id},
        )
        provenance = item.sample.metadata.provenance.model_copy(
            update={
                "source_reference": StoredRecordReference(
                    reference=source_object,
                    schema_version=1,
                )
            }
        )
        metadata = item.sample.metadata.model_copy(
            update={"provenance": provenance}
        )
        selected_sample = item.sample.model_copy(update={"metadata": metadata})
        inputs.append(item.model_copy(update={"sample": selected_sample}))
    return batch_request.model_copy(update={"inputs": tuple(inputs)})


async def publish_batch(
    root: Path,
    *,
    placement: RecordPlacement = RecordPlacement.BUNDLE_LOCAL,
    projections: tuple[ProjectionKind, ...] = tuple(ProjectionKind),
    count: int = 1,
    cache_store: BatchStore | None = None,
    object_store: ObjectStore | None = None,
):  # type: ignore[no-untyped-def]
    publication = ArtifactBundlePublication.allocate(root, prefix="evaluation")
    if object_store is None:
        object_store = ObjectStore(MemoryBackend())
    batch_request = await stored_source_request(
        request(count, projections=projections).model_copy(
            update={"record_placement": placement}
        ),
        object_store=object_store,
    )
    execution_cache = cache(cache_store or BatchStore(), resident=1)
    try:
        result = await evaluate_batch(
            batch_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=object_store,
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()
    return result, object_store
