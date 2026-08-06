from __future__ import annotations

import logging
from pathlib import Path

import pytest
from dr_store import (
    CacheHit,
    MemoryBackend,
    ObjectStore,
    RecordCache,
    SqliteRecordCache,
)

from dr_code.caching import (
    preprocessing_trace_cache_key,
    run_preprocessing_cached,
)
from dr_code.preprocessing import (
    BoundPreprocessingRunner,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    PreprocessingDefinition,
    StepName,
    StepSpec,
    bind_external_preprocessing,
    bind_preprocessing,
)
from dr_code.trace import (
    Absent,
    Artifact,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    is_absent,
    serialize_trace,
)

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"
_PROSE = "Just an explanation, no code at all.\n"


class _CountingRunner:
    def __init__(self, runner: BoundPreprocessingRunner) -> None:
        self._runner = runner
        self.runs = 0

    @property
    def producer(self) -> TraceProducer:
        return self._runner.producer

    def run(self, input_value: Artifact) -> Trace:
        self.runs += 1
        return self._runner.run(input_value)


def _memory_cache() -> RecordCache:
    return RecordCache(ObjectStore(MemoryBackend()))


class _HitCache:
    def __init__(self, record: dict[str, object]) -> None:
        self._record = record

    def get(self, key: str, *, schema: str) -> CacheHit:
        return CacheHit(record=self._record)

    def put(self, key: str, schema: str, record: object) -> None:
        pass


def _cache_for_trace(trace: Trace) -> _HitCache:
    return _HitCache(serialize_trace(trace).model_dump(mode="json"))


def _bound() -> _CountingRunner:
    return _CountingRunner(
        bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    )


def _assert_same_trace(restored: Trace, fresh: Trace) -> None:
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

    run_preprocessing_cached(_FENCED, runner, cache)
    run_preprocessing_cached(_PROSE, runner, cache)

    assert runner.runs == 2


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


def _cache_warnings(
    caplog: pytest.LogCaptureFixture,
) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == run_preprocessing_cached.__module__
        and record.levelno == logging.WARNING
    ]


@pytest.mark.parametrize(
    "cache",
    [_ReadFailingCache(), _WriteFailingCache(), _HitCache({})],
    ids=["read-failure", "write-failure", "invalid-record"],
)
def test_cache_failure_runs_fresh_and_logs(
    cache: RecordCache,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _bound()

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    assert trace.value("input") == TextArtifact(text=_FENCED)
    warnings = _cache_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None


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
    cache = _cache_for_trace(wrong)

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    assert trace.value("input") == TextArtifact(text=_FENCED)
    warnings = _cache_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].exc_info is None


def test_hit_with_wrong_producer_runs_fresh(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _bound()
    wrong = bind_external_preprocessing(_external("wrong")).run(
        TextArtifact(text=_FENCED)
    )
    cache = _cache_for_trace(wrong)

    with caplog.at_level(logging.WARNING):
        trace = run_preprocessing_cached(_FENCED, runner, cache)

    assert runner.runs == 1
    assert trace.producer == runner.producer
    warnings = _cache_warnings(caplog)
    assert len(warnings) == 1
    assert warnings[0].exc_info is None


def test_sqlite_cache_serves_a_hit_from_a_reopened_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "traces.sqlite3"
    writer = _bound()
    with SqliteRecordCache(database) as cache:
        fresh = run_preprocessing_cached(_FENCED, writer, cache)

    reader = _bound()
    with SqliteRecordCache(database) as cache:
        hit = run_preprocessing_cached(_FENCED, reader, cache)

    assert writer.runs == 1
    assert reader.runs == 0
    _assert_same_trace(hit, fresh)
