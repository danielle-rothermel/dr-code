from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import pytest
from dr_exec import AutoPoolCapacity, ExecutionPoolConfig
from dr_serialize import canonical_json_bytes
from dr_serialize import Sha256Digest
from dr_store import ArtifactBundlePublication, ObjectStore
from dr_store import MemoryBackend
from pydantic import ValidationError

from dr_code.evaluation import (
    BundleRecordReference,
    PreprocessMode,
    ProjectionKind,
    RecordPlacement,
    EvaluatedSampleRecord,
    read_eval_projection,
    restore_eval_attempt,
    ReusedCandidateProvenance,
    audit_eval_bundle,
    evaluate_batch,
)

from _executor_stubs import importable_json_executor

from ._batch_builders import BatchStore, cache, request
from ._bundle_builders import publish_batch, read_limits

pytestmark = pytest.mark.asyncio


async def test_selected_projection_reads_only_its_self_bound_artifact(
    tmp_path: Path,
) -> None:
    result, _ = await publish_batch(tmp_path)
    assert result.bundle_path is not None
    (result.bundle_path / "evaluation-attempt.json").write_bytes(b"tampered")
    header, rows = read_eval_projection(
        result.bundle_path,
        ProjectionKind.EVAL_SAMPLES,
        limits=read_limits(),
    )
    assert header.source_attempt == result.attempt.identity
    assert len(rows) == 1
    assert rows[0].source_attempt == result.attempt.identity


@pytest.mark.parametrize(
    "placement",
    [RecordPlacement.BUNDLE_LOCAL, RecordPlacement.OBJECT_STORE],
)
async def test_restoration_consumes_required_records_without_preaudit(
    tmp_path: Path,
    placement: RecordPlacement,
) -> None:
    result, object_store = await publish_batch(
        tmp_path, placement=placement, projections=()
    )
    assert result.bundle_path is not None
    restored = await restore_eval_attempt(
        result.bundle_path,
        object_store=object_store,
        limits=read_limits(),
    )
    assert restored.attempt == result.attempt
    assert [record.sample.identity for record in restored.samples] == [
        member.sample for member in result.attempt.members
    ]


async def test_restoration_enforces_nested_reference_depth(
    tmp_path: Path,
) -> None:
    result, object_store = await publish_batch(tmp_path, projections=())
    assert result.bundle_path is not None
    limits = read_limits().model_copy(update={"max_reference_depth": 1})

    with pytest.raises(ValueError, match="max_reference_depth"):
        await restore_eval_attempt(
            result.bundle_path,
            object_store=object_store,
            limits=limits,
        )


async def test_publication_rejects_an_unclosed_nested_bundle_reference(
    tmp_path: Path,
) -> None:
    batch_request = request(
        preprocess_mode=PreprocessMode.IN_PROCESS, projections=()
    )
    item = batch_request.inputs[0]
    provenance = item.data.sample.metadata.provenance.model_copy(
        update={
            "source_reference": BundleRecordReference(
                artifact_name="inputs-0.json",
                record_index=0,
                record_sha256=Sha256Digest("a" * 64),
                schema="tests/input",
                schema_version=1,
            )
        }
    )
    selected_sample = item.data.sample.model_copy(
        update={
            "metadata": item.data.sample.metadata.model_copy(
                update={"provenance": provenance}
            )
        }
    )
    batch_request = batch_request.model_copy(
        update={
            "inputs": (
                item.model_copy(
                    update={
                        "data": item.data.model_copy(
                            update={"sample": selected_sample}
                        )
                    }
                ),
            )
        }
    )
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(ValueError, match="not closed by the bundle"):
            await evaluate_batch(
                batch_request,
                executor=importable_json_executor(),
                execution_cache=execution_cache,
                object_store=None,
                publication=ArtifactBundlePublication.allocate(
                    tmp_path, prefix="unclosed"
                ),
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
            )
    finally:
        await execution_cache.close()


async def test_restoration_resolves_reused_execution_source_graph(
    tmp_path: Path,
) -> None:
    shared_cache = BatchStore()
    object_store = ObjectStore(MemoryBackend())
    first_root = tmp_path / "first"
    first_root.mkdir()
    await publish_batch(
        first_root,
        placement=RecordPlacement.OBJECT_STORE,
        projections=(),
        cache_store=shared_cache,
        object_store=object_store,
    )
    second_root = tmp_path / "second"
    second_root.mkdir()
    second, _ = await publish_batch(
        second_root,
        projections=(),
        cache_store=shared_cache,
        object_store=object_store,
    )
    assert second.bundle_path is not None

    restored = await restore_eval_attempt(
        second.bundle_path,
        object_store=object_store,
        limits=read_limits(),
    )
    sample = restored.samples[0]
    assert isinstance(sample, EvaluatedSampleRecord)
    execution = sample.executions[0]
    assert isinstance(execution.provenance, ReusedCandidateProvenance)
    audit = await audit_eval_bundle(
        second.bundle_path,
        object_store=object_store,
        limits=read_limits(),
    )
    assert audit.object_read_count > 0

    shallow = read_limits().model_copy(update={"max_reference_depth": 2})
    with pytest.raises(ValueError, match="max_reference_depth"):
        await restore_eval_attempt(
            second.bundle_path,
            object_store=object_store,
            limits=shallow,
        )


async def test_stored_records_are_resolved_strictly_sequentially(
    tmp_path: Path,
) -> None:
    result, object_store = await publish_batch(
        tmp_path,
        placement=RecordPlacement.OBJECT_STORE,
        projections=(),
        count=2,
    )
    assert result.bundle_path is not None

    class GatedObjectStore(ObjectStore):
        def __init__(self, wrapped: ObjectStore) -> None:
            self.wrapped = wrapped
            self.started: asyncio.Queue[asyncio.Event] = asyncio.Queue()

        async def get(self, reference):  # type: ignore[no-untyped-def]
            release = asyncio.Event()
            await self.started.put(release)
            await release.wait()
            return await self.wrapped.get(reference)

    gated = GatedObjectStore(object_store)
    restoring = asyncio.create_task(
        restore_eval_attempt(
            result.bundle_path,
            object_store=gated,
            limits=read_limits(),
        )
    )
    for _ in range(4):
        release = await gated.started.get()
        assert gated.started.empty()
        release.set()
    restored = await restoring
    assert len(restored.samples) == 2


async def test_selected_projection_rejects_unsupported_and_unbound_payloads(
    tmp_path: Path,
) -> None:
    result, _ = await publish_batch(
        tmp_path,
        projections=(ProjectionKind.EVAL_SAMPLES,),
    )
    assert result.bundle_path is not None
    header, rows = read_eval_projection(
        result.bundle_path,
        ProjectionKind.EVAL_SAMPLES,
        limits=read_limits(),
    )
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="projection-unbound"
    )
    writer = publication.open_artifact("projection-evaluation-samples.jsonl")
    other_attempt = header.source_attempt.model_copy(
        update={"attempt_id": UUID(int=2)}
    )
    writer.write(
        canonical_json_bytes(header.model_dump(mode="json"))
        + b"\n"
        + canonical_json_bytes(
            rows[0]
            .model_copy(update={"source_attempt": other_attempt})
            .model_dump(mode="json")
        )
        + b"\n"
    )
    writer.finalize()
    publication.publish({})
    with pytest.raises(ValueError, match="source attempt"):
        read_eval_projection(
            publication.path,
            ProjectionKind.EVAL_SAMPLES,
            limits=read_limits(),
        )

    unsupported = ArtifactBundlePublication.allocate(
        tmp_path, prefix="projection-unsupported"
    )
    writer = unsupported.open_artifact("projection-evaluation-samples.jsonl")
    wire = header.model_dump(mode="json")
    wire["schema_version"] = 3
    writer.write(canonical_json_bytes(wire) + b"\n")
    writer.finalize()
    unsupported.publish({})
    with pytest.raises(ValidationError):
        read_eval_projection(
            unsupported.path,
            ProjectionKind.EVAL_SAMPLES,
            limits=read_limits(),
        )
