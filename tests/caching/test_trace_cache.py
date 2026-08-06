from __future__ import annotations

import logging

import pytest
from dr_store import MemoryBackend, ObjectStore, RecordCache

from dr_code.caching import (
    TRACE_RECORD_SCHEMA,
    open_sqlite_record_cache,
    preprocessing_trace_cache_key,
    run_preprocessing_cached,
)
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    PreprocessingDefinition,
    StepName,
    StepSpec,
    bind_external_preprocessing,
    bind_preprocessing,
)
from dr_code.trace import (
    Absent,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    is_absent,
    serialize_trace,
)

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"
_PROSE = "Just an explanation, no code at all.\n"


class _CountingRunner:
    """Delegate to a bound runner while counting the runs it performs."""

    def __init__(self, runner) -> None:  # noqa: ANN001
        self._runner = runner
        self.runs = 0

    @property
    def producer(self):  # noqa: ANN201
        return self._runner.producer

    def run(self, input_value):  # noqa: ANN001, ANN201
        self.runs += 1
        return self._runner.run(input_value)


def _memory_cache() -> RecordCache:
    return RecordCache(ObjectStore(MemoryBackend()))


def _cache_bound_to_trace(key: str, trace) -> RecordCache:  # noqa: ANN001
    store = ObjectStore(MemoryBackend())
    reference, _ = store.put(
        TRACE_RECORD_SCHEMA,
        serialize_trace(trace).model_dump(mode="json"),
    )
    store.bind(key, reference)
    return RecordCache(store)


def _bound() -> _CountingRunner:
    return _CountingRunner(
        bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    )


def _assert_same_trace(restored, fresh) -> None:  # noqa: ANN001
    assert dict(restored.values) == dict(fresh.values)
    assert dict(restored.step_facts) == dict(fresh.step_facts)
    assert restored.producer == fresh.producer


def test_hit_returns_the_trace_a_fresh_run_produces() -> None:
    cache = _memory_cache()
    runner = _bound()

    fresh = run_preprocessing_cached(_FENCED, runner, cache)
    hit = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    _assert_same_trace(hit, fresh)
    output = hit.value("output")
    assert isinstance(output, InspectedCodeCandidateSetArtifact)
    assert output == fresh.value("output")


def test_absent_output_survives_the_cache() -> None:
    cache = _memory_cache()
    runner = _bound()

    fresh = run_preprocessing_cached(_PROSE, runner, cache)
    hit = run_preprocessing_cached(_PROSE, runner, cache)

    assert runner.runs == 1
    assert is_absent(fresh.value("output"))
    restored_output = hit.value("output")
    assert isinstance(restored_output, Absent)
    _assert_same_trace(hit, fresh)


def test_different_text_misses_and_runs_again() -> None:
    cache = _memory_cache()
    runner = _bound()

    first = run_preprocessing_cached(_FENCED, runner, cache)
    second = run_preprocessing_cached(_PROSE, runner, cache)

    assert runner.runs == 2
    assert dict(first.values) != dict(second.values)


def test_empty_cache_misses_and_returns_the_direct_run() -> None:
    cache = _memory_cache()
    runner = _bound()

    cached = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    _assert_same_trace(
        cached,
        bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION).run(
            cached.value("input")
        ),
    )


class _ReadFailingCache:
    def get(self, key: str, *, schema: str) -> None:
        raise OSError("read unavailable")

    def put(self, key: str, schema: str, record: object) -> None:
        pass


class _WriteFailingCache:
    def get(self, key: str, *, schema: str) -> None:
        return None

    def put(self, key: str, schema: str, record: object) -> None:
        raise OSError("write unavailable")


def test_read_failure_runs_fresh_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _bound()

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, _ReadFailingCache())

    assert runner.runs == 1
    assert trace.value("input") == TextArtifact(text=_FENCED)
    assert "cache read failed; running fresh" in caplog.text


def test_write_failure_returns_fresh_trace_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _bound()

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, _WriteFailingCache())

    assert runner.runs == 1
    assert trace.value("input") == TextArtifact(text=_FENCED)
    assert "cache write failed; returning fresh trace" in caplog.text


def _external(definition_id: str) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        definition_id=definition_id,
        version="0",
        steps=(
            StepSpec(
                instance_name="normalize_unicode",
                step=StepName.NORMALIZE_UNICODE,
            ),
        ),
    )


def test_key_separates_distinct_definition_coordinates() -> None:
    one = bind_external_preprocessing(_external("one"))
    other = bind_external_preprocessing(_external("other"))

    assert preprocessing_trace_cache_key(
        _FENCED, one
    ) != preprocessing_trace_cache_key(_FENCED, other)


def test_key_is_stable_across_equal_bindings() -> None:
    assert preprocessing_trace_cache_key(
        _FENCED, bind_external_preprocessing(_external("one"))
    ) == preprocessing_trace_cache_key(
        _FENCED, bind_external_preprocessing(_external("one"))
    )


def test_hit_with_wrong_input_runs_fresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _bound()
    wrong = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION).run(
        TextArtifact(text=_PROSE)
    )
    key = preprocessing_trace_cache_key(_FENCED, runner)
    cache = _cache_bound_to_trace(key, wrong)

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    assert trace.value("input") == TextArtifact(text=_FENCED)
    assert "cache entry has the wrong input" in caplog.text


def test_hit_with_wrong_producer_runs_fresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _bound()
    wrong = bind_external_preprocessing(_external("wrong")).run(
        TextArtifact(text=_FENCED)
    )
    key = preprocessing_trace_cache_key(_FENCED, runner)
    cache = _cache_bound_to_trace(key, wrong)

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    assert trace.producer == runner.producer
    assert "cache entry has the wrong producer" in caplog.text


def test_sqlite_cache_serves_a_hit_from_a_reopened_database(
    tmp_path,
) -> None:  # noqa: ANN001
    database = tmp_path / "traces.sqlite3"
    writer = _bound()
    fresh = run_preprocessing_cached(
        _FENCED, writer, open_sqlite_record_cache(database)
    )

    reader = _bound()
    hit = run_preprocessing_cached(
        _FENCED, reader, open_sqlite_record_cache(database)
    )

    assert writer.runs == 1
    assert reader.runs == 0
    _assert_same_trace(hit, fresh)
