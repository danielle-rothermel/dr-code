"""Unit tests for eval run lifecycle hardening."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from dr_queues import RunManifest, RunStageManifest

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.pipeline import lifecycle
from dr_code.pipeline.definition import build_eval_pipeline
from dr_code.pipeline.handlers import registry
from dr_code.pipeline.metadata import EvalRunAlreadySeededError, EvalSeedSource


class _RunStore:
    def __init__(self, expected_jobs: int = 0) -> None:
        self.expected_jobs = expected_jobs

    def expected_job_count(self, run_id: str) -> int:
        assert run_id == "run-1"
        return self.expected_jobs


class _MetadataStore:
    def __init__(self) -> None:
        self.init_calls = []
        self.seed_calls = []

    def record_init(self, **kwargs):
        self.init_calls.append(kwargs)

    def record_seed(self, **kwargs):
        self.seed_calls.append(kwargs)

    def get(self, run_id: str):
        assert run_id == "run-1"
        return SimpleNamespace(run_id=run_id)


def _record(sample_id: str = "sample-1") -> AttemptRecord:
    return AttemptRecord(
        sample_id=sample_id,
        run_id=None,
        task_id="HumanEval/0",
        entry_point="fn",
        decoder_input="desc",
        raw_output="code",
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )


def _manifest() -> RunManifest:
    pipeline = build_eval_pipeline(registry)
    return RunManifest(
        run_id="run-1",
        pipeline_definition=pipeline.definition,
        queue_prefix="run.run-1",
        stages=[
            RunStageManifest(
                name="parse",
                step_index=0,
                handler_key="parse_attempt",
                input_queue="run.run-1.s1.pending",
                output_queue="run.run-1.s1.completed",
                default_workers=2,
            ),
            RunStageManifest(
                name="test",
                step_index=1,
                handler_key="run_tests",
                input_queue="run.run-1.s1.completed",
                output_queue="run.run-1.s2.completed",
                default_workers=3,
            ),
        ],
    )


def test_init_eval_run_records_metadata(monkeypatch) -> None:
    metadata_store = _MetadataStore()
    manifest = _manifest()

    monkeypatch.setattr(
        lifecycle,
        "setup_run_queues",
        lambda **kwargs: manifest,
    )

    result = lifecycle.init_eval_run(
        run_id="run-1",
        workers="parse=2,test=3",
        run_store=SimpleNamespace(),
        metadata_store=metadata_store,
    )

    assert result.run_id == "run-1"
    assert result.workers_by_stage == {"parse": 2, "test": 3}
    assert metadata_store.init_calls[0]["run_id"] == "run-1"
    assert metadata_store.init_calls[0]["metadata"].worker_spec == "parse=2,test=3"


def test_seed_eval_run_refuses_existing_seed_before_publish(monkeypatch) -> None:
    seed_calls = []
    monkeypatch.setattr(lifecycle, "seed_run", lambda *args, **kwargs: seed_calls.append(kwargs))

    with pytest.raises(EvalRunAlreadySeededError):
        lifecycle.seed_eval_run(
            [_record()],
            run_id="run-1",
            run_store=_RunStore(expected_jobs=1),
            metadata_store=_MetadataStore(),
        )

    assert seed_calls == []


def test_seed_eval_run_records_seed_metadata(monkeypatch) -> None:
    seed_calls = []
    metadata_store = _MetadataStore()
    monkeypatch.setattr(lifecycle, "attach_run_queues", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(lifecycle, "seed_run", lambda *args, **kwargs: seed_calls.append(kwargs))

    result = lifecycle.seed_eval_run(
        [_record()],
        run_id="run-1",
        run_store=_RunStore(),
        metadata_store=metadata_store,
    )

    assert result.expected_jobs == 1
    assert result.metadata.source == EvalSeedSource.IN_MEMORY
    assert seed_calls
    assert metadata_store.seed_calls[0]["run_id"] == "run-1"
    assert metadata_store.seed_calls[0]["metadata"].expected_jobs == 1
