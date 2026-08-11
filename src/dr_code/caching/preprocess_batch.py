from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, TypeVar
from uuid import uuid4

from dr_exec import (
    AutoPoolCapacity,
    ExecutionPoolConfig,
    ExecutionSubmission,
    FixedPoolCapacity,
    ImportableJsonExecutor,
    JobId,
)
from dr_store import RecordCache

from dr_code.caching.execution_cache import BatchRecordStore
from dr_code.caching.trace_cache import (
    preprocessing_trace_cache_key,
    trace_from_cached_record,
)
from dr_code.caching.trace_window_cache import WindowedTraceCache
from dr_code.preprocessing.definition import PreprocessingDefinition
from dr_code.preprocessing.execution import (
    PreprocessTextExecutionError,
    build_preprocess_text_job,
    parse_preprocess_text_result,
    preprocess_text_request,
)
from dr_code.preprocessing.runner import bind_preprocessing
from dr_code.trace import TextArtifact, Trace, serialize_trace

_T = TypeVar("_T")

DEFAULT_WORKER_COUNT: Final = 16
_RESIDENT_MULTIPLIER: Final = 4


@dataclass(frozen=True, slots=True)
class PreprocessBatchLimits:
    max_resident_entries: int
    max_pending_checkpoint_entries: int
    max_prefetch_keys: int
    max_admitted_jobs: int


@dataclass(frozen=True, slots=True)
class _PreprocessWork:
    text: str
    cache_key: str
    index: int


def resolved_pool_worker_count(pool_config: ExecutionPoolConfig) -> int:
    capacity = pool_config.capacity
    if isinstance(capacity, FixedPoolCapacity):
        return capacity.max_active_jobs
    if isinstance(capacity, AutoPoolCapacity):
        from dr_exec.scheduling.pool import usable_cpu_count

        return usable_cpu_count()
    raise TypeError(f"unsupported pool capacity: {type(capacity)!r}")


def default_preprocess_batch_limits(
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
) -> PreprocessBatchLimits:
    resident = max(worker_count * _RESIDENT_MULTIPLIER, worker_count)
    return PreprocessBatchLimits(
        max_resident_entries=resident,
        max_pending_checkpoint_entries=resident,
        max_prefetch_keys=resident,
        max_admitted_jobs=worker_count,
    )


async def preprocess_batch(
    texts: Sequence[str],
    *,
    definition: PreprocessingDefinition,
    store: BatchRecordStore,
    pool_config: ExecutionPoolConfig,
    limits: PreprocessBatchLimits | None = None,
    on_trace_completed: Callable[[str, Trace], None] | None = None,
) -> dict[str, Trace]:
    """Preprocess distinct texts with bounded cache windows and worker pool."""

    effective_limits = limits or default_preprocess_batch_limits()
    runner = bind_preprocessing(definition)
    distinct_texts = list(dict.fromkeys(texts))
    if not distinct_texts:
        return {}

    work_by_key = {
        preprocessing_trace_cache_key(text, runner): text
        for text in distinct_texts
    }
    cache_keys = list(work_by_key)

    results: dict[str, Trace] = {}
    executor = ImportableJsonExecutor()
    async with WindowedTraceCache(
        store,
        max_resident_entries=effective_limits.max_resident_entries,
        max_pending_checkpoint_entries=(
            effective_limits.max_pending_checkpoint_entries
        ),
    ) as trace_cache:
        for key_window in _windows(
            cache_keys, effective_limits.max_prefetch_keys
        ):
            try:
                await trace_cache.prefetch(key_window)
                misses: list[_PreprocessWork] = []
                for index, cache_key in enumerate(key_window):
                    text = work_by_key[cache_key]
                    input_value = TextArtifact(text=text)
                    cached = trace_cache.get(cache_key)
                    if cached is None:
                        misses.append(
                            _PreprocessWork(
                                text=text,
                                cache_key=cache_key,
                                index=index,
                            )
                        )
                        continue
                    restored = trace_from_cached_record(
                        cached,
                        input_value=input_value,
                        runner=runner,
                    )
                    if restored is None:
                        misses.append(
                            _PreprocessWork(
                                text=text,
                                cache_key=cache_key,
                                index=index,
                            )
                        )
                    else:
                        results[text] = restored
                        if on_trace_completed is not None:
                            on_trace_completed(text, restored)

                for admission_window in _windows(
                    misses,
                    effective_limits.max_admitted_jobs,
                ):
                    async with executor.open_pool(config=pool_config) as pool:
                        async for completion in pool.run_stream(
                            _submissions(admission_window, definition)
                        ):
                            item = completion.context
                            try:
                                trace = parse_preprocess_text_result(
                                    completion.completed_execution
                                )
                            except PreprocessTextExecutionError:
                                continue
                            results[item.text] = trace
                            if on_trace_completed is not None:
                                on_trace_completed(item.text, trace)
                            await trace_cache.put(
                                item.cache_key,
                                serialize_trace(trace),
                            )
            finally:
                for cache_key in key_window:
                    trace_cache.discard(cache_key)

    return results


async def preprocess_batch_cached(
    texts: Sequence[str],
    *,
    definition: PreprocessingDefinition,
    cache: RecordCache,
    pool_config: ExecutionPoolConfig | None = None,
    limits: PreprocessBatchLimits | None = None,
) -> dict[str, Trace]:
    """Preprocess texts through one persistent record cache."""

    worker_count = DEFAULT_WORKER_COUNT
    if pool_config is not None:
        worker_count = resolved_pool_worker_count(pool_config)
    effective_pool_config = pool_config or ExecutionPoolConfig(
        capacity=FixedPoolCapacity(max_active_jobs=worker_count)
    )
    return await preprocess_batch(
        texts,
        definition=definition,
        store=cache,
        pool_config=effective_pool_config,
        limits=limits
        or default_preprocess_batch_limits(worker_count=worker_count),
    )


async def _submissions(
    work: Sequence[_PreprocessWork],
    definition: PreprocessingDefinition,
) -> AsyncIterator[ExecutionSubmission[_PreprocessWork]]:
    for item in work:
        request = preprocess_text_request(item.text, definition)
        yield ExecutionSubmission(
            job=build_preprocess_text_job(JobId(uuid4()), request),
            context=item,
        )


def _windows(values: Sequence[_T], size: int) -> Iterable[tuple[_T, ...]]:
    if size < 1:
        raise ValueError("window size must be positive")
    return (
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
    )


__all__ = [
    "DEFAULT_WORKER_COUNT",
    "PreprocessBatchLimits",
    "default_preprocess_batch_limits",
    "preprocess_batch",
    "preprocess_batch_cached",
    "resolved_pool_worker_count",
]
