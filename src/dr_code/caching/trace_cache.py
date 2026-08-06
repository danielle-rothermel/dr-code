from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

from dr_store import derive_cache_key

from dr_code.trace import (
    INPUT_KEY,
    TRACE_SCHEMA_VERSION,
    SerializedTrace,
    TextArtifact,
    deserialize_trace,
    serialize_trace,
)

if TYPE_CHECKING:
    from dr_store import RecordCache

    from dr_code.preprocessing import BoundPreprocessingRunner
    from dr_code.trace import Trace

TRACE_CACHE_NAMESPACE: Final = "dr-code/preprocessing-trace"
TRACE_RECORD_SCHEMA: Final = f"dr-code/serialized-trace@{TRACE_SCHEMA_VERSION}"
_LOGGER = logging.getLogger(__name__)


def preprocessing_trace_cache_key(
    text: str,
    runner: BoundPreprocessingRunner,
) -> str:
    """Key one raw text against one bound definition's resolved coordinate.

    The runner's producer already carries the definition coordinate with its
    resolved component versions and settings, so a changed step, setting, or
    trace schema version derives a different key instead of reusing a trace
    that a different composition produced.
    """
    return derive_cache_key(
        TRACE_CACHE_NAMESPACE,
        {
            "text": text,
            "producer": runner.producer.model_dump(mode="json"),
            "trace_schema_version": TRACE_SCHEMA_VERSION,
        },
    )


def run_preprocessing_cached(
    text: str,
    runner: BoundPreprocessingRunner,
    cache: RecordCache,
) -> Trace:
    """Return the cached trace for ``text``, otherwise run and store one.

    A serialized trace is self-describing, so restoring one consults no
    registry and a hit differs from a miss only in cost. Cache faults and
    entries that do not describe this request are logged and treated as
    misses.
    """
    key = preprocessing_trace_cache_key(text, runner)
    input_value = TextArtifact(text=text)
    cached = _restore_cached_trace(
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
        cache.put(key, TRACE_RECORD_SCHEMA, record)
    except Exception:
        # Cache implementations are optional infrastructure. Preserve the
        # successful computation while retaining the failure traceback.
        _LOGGER.warning(
            "preprocessing trace cache write failed; returning fresh trace",
            exc_info=True,
        )
    return trace


def _restore_cached_trace(
    *,
    key: str,
    input_value: TextArtifact,
    runner: BoundPreprocessingRunner,
    cache: RecordCache,
) -> Trace | None:
    try:
        hit = cache.get(key, schema=TRACE_RECORD_SCHEMA)
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
    "TRACE_CACHE_NAMESPACE",
    "TRACE_RECORD_SCHEMA",
    "preprocessing_trace_cache_key",
    "run_preprocessing_cached",
]
