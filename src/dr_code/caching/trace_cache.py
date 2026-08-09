from __future__ import annotations

import logging
from typing import Final

from dr_store import RecordCache, derive_cache_key

from dr_code.preprocessing import BoundPreprocessingRunner
from dr_code.trace import (
    INPUT_KEY,
    TRACE_SCHEMA_VERSION,
    SerializedTrace,
    TextArtifact,
    Trace,
    deserialize_trace,
    serialize_trace,
)

_TRACE_CACHE_NAMESPACE: Final = "dr-code/preprocessing-trace"
_TRACE_RECORD_SCHEMA: Final = (
    f"dr-code/serialized-trace@{TRACE_SCHEMA_VERSION}"
)
_LOGGER = logging.getLogger(__name__)


def preprocessing_trace_cache_key(
    text: str,
    runner: BoundPreprocessingRunner,
) -> str:
    """Key raw text, its resolved producer coordinate, and trace schema."""
    return derive_cache_key(
        _TRACE_CACHE_NAMESPACE,
        {
            "text": text,
            "producer": runner.producer.model_dump(mode="json"),
            "trace_schema_version": TRACE_SCHEMA_VERSION,
        },
    )


async def run_preprocessing_cached(
    text: str,
    runner: BoundPreprocessingRunner,
    cache: RecordCache,
) -> Trace:
    """Return a matching cached trace, or run and best-effort store one.

    Callers invalidate caches when behavior changes without a producer
    coordinate change.
    """
    key = preprocessing_trace_cache_key(text, runner)
    input_value = TextArtifact(text=text)
    cached = await _restore_cached_trace(
        key=key,
        input_value=input_value,
        runner=runner,
        cache=cache,
    )
    if cached is not None:
        return cached

    trace = runner.run(input_value)
    record = serialize_trace(trace).model_dump(mode="json")
    try:
        await cache.put(key, _TRACE_RECORD_SCHEMA, record)
    except Exception:
        # Cache implementations are optional infrastructure. Preserve the
        # successful computation while retaining the failure traceback.
        _LOGGER.warning(
            "preprocessing trace cache write failed; returning fresh trace",
            exc_info=True,
        )
    return trace


async def _restore_cached_trace(
    *,
    key: str,
    input_value: TextArtifact,
    runner: BoundPreprocessingRunner,
    cache: RecordCache,
) -> Trace | None:
    try:
        hit = await cache.get(key, schema=_TRACE_RECORD_SCHEMA)
    except Exception:
        _LOGGER.warning(
            "preprocessing trace cache read failed; running fresh",
            exc_info=True,
        )
        return None
    if hit is None:
        return None

    try:
        restored = deserialize_trace(
            SerializedTrace.model_validate(hit.record)
        )
    except Exception:
        _LOGGER.warning(
            "invalid preprocessing trace cache entry; running fresh",
            exc_info=True,
        )
        return None
    if restored.producer != runner.producer:
        _LOGGER.warning(
            "preprocessing trace cache entry has the wrong producer; "
            "running fresh"
        )
        return None
    if restored.value(INPUT_KEY) != input_value:
        _LOGGER.warning(
            "preprocessing trace cache entry has the wrong input; running fresh"
        )
        return None
    return restored


__all__ = [
    "preprocessing_trace_cache_key",
    "run_preprocessing_cached",
]
