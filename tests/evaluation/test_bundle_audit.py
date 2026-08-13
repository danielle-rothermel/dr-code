from __future__ import annotations

from pathlib import Path

import pytest
from dr_store import BundleVerificationError

from dr_code.evaluation import (
    RecordPlacement,
    audit_eval_bundle,
)

from ._bundle_builders import publish_batch, read_limits

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "placement",
    [RecordPlacement.BUNDLE_LOCAL, RecordPlacement.OBJECT_STORE],
)
async def test_audit_validates_closed_domain_graph_and_counts_object_reads(
    tmp_path: Path,
    placement: RecordPlacement,
) -> None:
    result, object_store = await publish_batch(tmp_path, placement=placement)
    assert result.bundle_path is not None
    audit = await audit_eval_bundle(
        result.bundle_path,
        object_store=object_store,
        limits=read_limits(),
    )
    assert audit.attempt == result.attempt.identity
    assert audit.sample_record_count == 1
    assert audit.object_read_count == (
        1 if placement is RecordPlacement.BUNDLE_LOCAL else 2
    )
    assert audit.artifact_count == 7


async def test_audit_rejects_tampered_artifact_before_domain_validation(
    tmp_path: Path,
) -> None:
    result, object_store = await publish_batch(tmp_path)
    assert result.bundle_path is not None
    projection = result.bundle_path / "projection-scores.jsonl"
    projection.write_bytes(projection.read_bytes() + b"tampered")
    with pytest.raises(BundleVerificationError):
        await audit_eval_bundle(
            result.bundle_path,
            object_store=object_store,
            limits=read_limits(),
        )
