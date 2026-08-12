from __future__ import annotations

import pytest
from dr_exec import ExecutionPoolConfig, FixedPoolCapacity

from _executor_stubs import importable_json_executor
from dr_code.caching.preprocess_batch import preprocess_batch
from dr_code.evaluation._batch import _evaluate_batch_assembly
from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION

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


async def test_preprocessed_trace_must_match_sample_raw_input() -> None:
    text_a = "def observed_load_count(_x):\n    return 1\n"
    text_b = "def other(_x):\n    return 2\n"
    batch_request = request(1, texts=(text_a,))
    traces = await preprocess_batch(
        (text_b,),
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        worker_count=1,
    )
    executor = importable_json_executor()
    execution_cache = cache(BatchStore(), resident=64)
    placement = MemoryPlacement()
    with pytest.raises(ValueError, match="preprocessed trace input"):
        await _evaluate_batch_assembly(
            batch_request,
            executor=executor,
            execution_cache=execution_cache,
            pool_config=ExecutionPoolConfig(
                capacity=FixedPoolCapacity(max_active_jobs=1)
            ),
            placement_sink=placement,
            preprocessed_traces={text_a: traces[text_b]},
        )
