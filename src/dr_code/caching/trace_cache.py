from __future__ import annotations

from typing import TYPE_CHECKING, Final

from dr_store import derive_cache_key

from dr_code.trace import (
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
    registry and a hit differs from a miss only in cost. Storage-level
    misses fall through to a fresh run.
    """
    key = preprocessing_trace_cache_key(text, runner)
    hit = cache.get(key, schema=TRACE_RECORD_SCHEMA)
    if hit is not None:
        return deserialize_trace(SerializedTrace.model_validate(hit.record))
    trace = runner.run(TextArtifact(text=text))
    cache.put(
        key,
        TRACE_RECORD_SCHEMA,
        serialize_trace(trace).model_dump(mode="json"),
    )
    return trace


__all__ = [
    "TRACE_CACHE_NAMESPACE",
    "TRACE_RECORD_SCHEMA",
    "preprocessing_trace_cache_key",
    "run_preprocessing_cached",
]
