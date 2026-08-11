from __future__ import annotations

import pytest
from dr_exec import ExecutionPoolConfig, FixedPoolCapacity

from _executor_stubs import importable_json_executor
from dr_code.evaluation._batch import _evaluate_batch_assembly

from ._batch_builders import BatchStore, MemoryPlacement, cache, request

pytestmark = pytest.mark.asyncio


async def test_evaluate_batch_prepares_samples_via_preprocess_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_request = request(2)
    observed_texts: list[str] = []
    original = __import__(
        "dr_code.caching.preprocess_batch",
        fromlist=["preprocess_batch"],
    ).preprocess_batch

    async def _spy_preprocess_batch(texts, **kwargs):  # noqa: ANN001
        observed_texts.extend(texts)
        return await original(texts, **kwargs)

    monkeypatch.setattr(
        "dr_code.evaluation._batch.preprocess_batch",
        _spy_preprocess_batch,
    )

    executor = importable_json_executor()
    execution_cache = cache(BatchStore(), resident=64)
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=executor,
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(
            capacity=FixedPoolCapacity(max_active_jobs=2)
        ),
        placement_sink=placement,
    )

    assert len(observed_texts) == 2
