from __future__ import annotations

import json

import pytest
from dr_serialize import StrictJsonError

from dr_code.classifier.aggregation import RepeatPhase
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
    load_artifact,
    load_records,
    write_records_atomic,
)
from dr_code.classifier.taxonomy import FailureFamily


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
    write_records_atomic(
        path,
        experiment,
        tuple(reversed(load_records(path))),
    )
    artifact = load_artifact(path)
    assert artifact is not None
    assert artifact.header.artifact_version == "failure-classifications-v4"
    assert artifact.header.schema_version == 4
    assert artifact.header.experiment_identity == experiment_identity(
        experiment
    )
    assert artifact.records[0].repeats[0].model_dump(mode="json") == {
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


def test_jsonl_rejects_duplicate_and_corrupt_loaded_records(tmp_path) -> None:
    path = tmp_path / "details.jsonl"
    experiment = _experiment()
    write_records_atomic(path, experiment, (_record(experiment),))
    header, line = path.read_text().splitlines()
    path.write_text(f"{header}\n{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_records(path)

    payload = json.loads(line)
    payload["unexpected"] = True
    path.write_text(f"{header}\n{json.dumps(payload)}\n")
    with pytest.raises(ValueError, match="invalid"):
        load_records(path)


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
