from __future__ import annotations

import json
import os
import weakref
from collections.abc import Iterator

import pytest
from dr_serialize import StrictJsonError

from dr_code.classifier.aggregation import RepeatPhase
import dr_code.classifier.records as records_module
from dr_code.classifier.records import (
    AggregateRecord,
    ClassifierConfigRecord,
    ClassifierExperimentRecord,
    ItemIdentityRecord,
    ItemRecord,
    RepeatRecord,
    ResumeIdentityRecord,
    RunScopeRecord,
    SelectionPolicyRecord,
    experiment_identity,
    write_records_atomic,
)
from dr_code.classifier.taxonomy import FailureFamily
from classifier.helpers import capture_artifact_for_test


def _experiment(
    *, with_evaluation: bool = False
) -> ClassifierExperimentRecord:
    return ClassifierExperimentRecord(
        run=RunScopeRecord(
            run_id="run",
            dataset_id="org/data",
            corpus_sha256="a" * 64,
            preprocessing_manifest_sha256="b" * 64,
            preprocessing_identity="c" * 64,
            preprocessing_schema_version=3,
            definition_id="definition",
            definition_version="1",
            definition_identity="d" * 64,
            evaluation_manifest_sha256="e" * 64 if with_evaluation else None,
            evaluation_generation_id="generation" if with_evaluation else None,
            evaluation_pointer_sha256="f" * 64 if with_evaluation else None,
            evaluation_identity="1" * 64 if with_evaluation else None,
        ),
        config=ClassifierConfigRecord(
            artifact_version="failure-classifications-v4",
            schema_version=4,
            extraction_version="extraction",
            aggregation_version="aggregation",
            taxonomy_version="taxonomy",
            taxonomy_identity="2" * 64,
            prompt_version="prompt",
            prompt_template_version="template",
            prompt_template_identity="3" * 64,
            prompt_max_evidence_chars=16_000,
            prompt_max_input_chars=12_000,
            prompt_max_task_context_chars=2_000,
            prompt_max_metadata_chars=512,
            prompt_correction_attempts=1,
            provider="provider",
            model="model",
            lane_policy_identity="4" * 64,
            lane_adapter="adapter",
            lane_executable=None,
            lane_timeout_seconds=None,
            repeats=1,
        ),
        selection=SelectionPolicyRecord(parse_limit=3, test_limit=None),
    )


def _record(
    experiment: ClassifierExperimentRecord,
    sample_id: str = "sample",
) -> ItemRecord:
    return ItemRecord(
        identity=ResumeIdentityRecord(
            experiment_identity=experiment_identity(experiment),
            repeats=experiment.config.repeats,
            item=ItemIdentityRecord(
                family=FailureFamily.PARSE,
                sample_id=sample_id,
                candidate_id=None,
                evaluation_key=None,
                task_id="Task",
                task_identity="5" * 64,
                rendered_input_sha256="4" * 64,
            ),
        ),
        aggregate=AggregateRecord(
            label="other",
            agreement=1.0,
            tie=False,
            successful_repeats=1,
            failed_repeats=0,
            label_counts={"other": 1},
        ),
        repeats=(
            RepeatRecord(
                index=0,
                label="other",
                rationale="reason",
                failure=None,
                phase=RepeatPhase.PRIMARY,
                attempt=1,
                corrected=False,
                primary_validation_failure=None,
            ),
        ),
    )


def test_jsonl_is_canonical_sorted_utf8_and_round_trips(tmp_path) -> None:
    path = tmp_path / "details.jsonl"
    experiment = _experiment()
    write_records_atomic(
        path,
        experiment,
        (_record(experiment, "z"), _record(experiment, "é")),
    )
    first = path.read_bytes()
    _, records = capture_artifact_for_test(path)
    write_records_atomic(
        path,
        experiment,
        reversed(records),
    )
    header, records = capture_artifact_for_test(path)
    assert header.artifact_version == "failure-classifications-v4"
    assert header.schema_version == 4
    assert header.experiment_identity == experiment_identity(experiment)
    assert records[0].repeats[0].model_dump(mode="json") == {
        "attempt": 1,
        "corrected": False,
        "failure": None,
        "index": 0,
        "label": "other",
        "phase": "primary",
        "primary_validation_failure": None,
        "rationale": "reason",
    }
    assert path.read_bytes() == first
    assert b"\\u00e9" not in first
    assert b": " not in first


def test_jsonl_rejects_duplicate_before_replacing_existing_bytes(
    tmp_path,
) -> None:
    path = tmp_path / "details.jsonl"
    experiment = _experiment()
    record = _record(experiment)
    write_records_atomic(path, experiment, (record,))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="duplicate"):
        write_records_atomic(path, experiment, (record, record))

    assert path.read_bytes() == before


@pytest.mark.parametrize("records_are_sorted", [False, True])
def test_atomic_writer_never_closes_reused_descriptor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    records_are_sorted: bool,
) -> None:
    path = tmp_path / "details.jsonl"
    experiment = _experiment()
    sentinel_descriptors: list[int] = []
    real_replace = records_module.os.replace

    def replace_and_reuse_descriptor(source, destination) -> None:
        real_replace(source, destination)
        sentinel_descriptors.append(os.open(os.devnull, os.O_RDONLY))

    monkeypatch.setattr(
        records_module.os, "replace", replace_and_reuse_descriptor
    )
    try:
        write_records_atomic(
            path,
            experiment,
            (_record(experiment),),
            records_are_sorted=records_are_sorted,
        )
        assert len(sentinel_descriptors) == 1
        os.fstat(sentinel_descriptors[0])
    finally:
        for descriptor in sentinel_descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_jsonl_rejects_duplicate_and_corrupt_loaded_records(tmp_path) -> None:
    path = tmp_path / "details.jsonl"
    experiment = _experiment()
    write_records_atomic(path, experiment, (_record(experiment),))
    header, line = path.read_text().splitlines()
    path.write_text(f"{header}\n{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate"):
        capture_artifact_for_test(path)

    payload = json.loads(line)
    payload["unexpected"] = True
    path.write_text(f"{header}\n{json.dumps(payload)}\n")
    with pytest.raises(ValueError, match="invalid"):
        capture_artifact_for_test(path)


def test_public_surface_excludes_materializing_artifact_apis() -> None:
    deleted_names = {
        "ClassificationArtifact",
        "_load_artifact_materialized",
        "load_artifact",
        "load_artifact_bytes",
        "load_records",
    }

    assert deleted_names.isdisjoint(records_module.__all__)
    assert all(not hasattr(records_module, name) for name in deleted_names)


@pytest.mark.parametrize("records_are_sorted", [False, True])
def test_atomic_writer_bounds_large_iterable_without_length_hint(
    tmp_path,
    records_are_sorted: bool,
) -> None:
    path = tmp_path / "details.jsonl"
    experiment = _experiment()
    record_count = 2_048

    class BoundedRecords:
        def __len__(self) -> int:
            raise AssertionError("writer requested an input length")

        def __length_hint__(self) -> int:
            raise AssertionError("writer requested an input length hint")

        def __iter__(self) -> Iterator[ItemRecord]:
            live: list[weakref.ReferenceType[ItemRecord]] = []
            for index in range(record_count):
                live[:] = [reference for reference in live if reference()]
                if len(live) >= 2:
                    raise AssertionError("writer retained unbounded records")
                record = _record(experiment, f"sample-{index:04d}")
                live.append(weakref.ref(record))
                yield record

    write_records_atomic(
        path,
        experiment,
        BoundedRecords(),
        records_are_sorted=records_are_sorted,
    )

    assert sum(1 for _line in path.open("rb")) == record_count + 1


def test_experiment_hash_preserves_null_coordinates_and_rejects_nonfinite() -> (
    None
):
    without_evaluation = _experiment()
    with_evaluation = _experiment(with_evaluation=True)

    payload = without_evaluation.model_dump(mode="json")
    assert "evaluation_manifest_sha256" in payload["run"]
    assert payload["run"]["evaluation_manifest_sha256"] is None
    assert experiment_identity(without_evaluation) != experiment_identity(
        with_evaluation
    )

    invalid_config = without_evaluation.config.model_copy(
        update={"lane_timeout_seconds": float("nan")}
    )
    invalid_experiment = without_evaluation.model_copy(
        update={"config": invalid_config}
    )
    with pytest.raises(StrictJsonError, match="non-finite"):
        experiment_identity(invalid_experiment)
