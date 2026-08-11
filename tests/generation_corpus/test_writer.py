from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from dr_code.generation_corpus import (
    CorpusWriter,
    BudgetMode,
    DatasetName,
    EncoderArtifactRecord,
    GenerationMode,
    GenerationRecord,
    LifecycleState,
    PromptFidelity,
    RequestRecord,
    SourceManifest,
    SourceRecord,
    Stage,
    TaskMaterialFidelity,
    TaskRecord,
    canonical_json,
    content_sha256,
)
from dr_code.generation_corpus.writer import (
    ENCODER_ARTIFACT_SCHEMA,
    GENERATION_SCHEMA,
    REQUEST_SCHEMA,
    SOURCE_RECORD_SCHEMA,
    TASK_SCHEMA,
)


def _source_manifest(tmp_path: Path) -> tuple[Path, SourceManifest]:
    payload = {
        "version": 1,
        "created_at": "2026-06-21T20:19:47Z",
        "output_dir": "/source/dump",
        "pools": [],
    }
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, SourceManifest.model_validate(payload)


def _source(source_id: str, *, stage: Stage) -> SourceRecord:
    output = (
        "description" if stage is Stage.ENCODER else "def f():\n    return 1\n"
    )
    request_prompt = "describe" if stage is Stage.ENCODER else "implement"
    return SourceRecord(
        source_record_id=source_id,
        dataset=DatasetName.HUMAN_EVAL,
        source_variant="canonical",
        data_sample_id="human_eval/HumanEval/0/gt_solution@abc",
        task_id="HumanEval/0",
        source_project="project",
        source_pool="pool",
        source_table="pool_pool_samples",
        source_file="pool.jsonl.gz",
        source_line_number=1,
        source_sample_id=source_id.rsplit(":", 1)[-1],
        sample_idx=0,
        run_id=None,
        stage=stage,
        lifecycle_state=(
            LifecycleState.ENCODER_ONLY
            if stage is Stage.ENCODER
            else LifecycleState.EVALUATED
        ),
        date="2026-05-10T10:00:00Z",
        date_kind="created_at",
        status="active",
        attempt_count=2,
        finish_reason="stop",
        response_finish_reason="stop",
        has_nonblank_output=True,
        output_text=output,
        key_values_json=canonical_json({}),
        request_json=canonical_json({"prompt": request_prompt}),
        response_json=canonical_json({"text": output}),
        metadata_json=canonical_json({}),
        hints_json=canonical_json({"output_kind": "code_text"}),
    )


def _task_hash() -> str:
    return content_sha256({"prompt": "def f(): ...", "tests": []})


def _generation() -> GenerationRecord:
    output = "def f():\n    return 1\n"
    return GenerationRecord(
        generation_id="generation-1",
        source_record_id="project:pool:decoder-1",
        dataset=DatasetName.HUMAN_EVAL,
        source_variant="canonical",
        task_record_id=_task_hash(),
        data_sample_id="human_eval/HumanEval/0/gt_solution@abc",
        task_id="HumanEval/0",
        generation_mode=GenerationMode.DIRECT,
        stage=Stage.DECODER,
        lifecycle_state=LifecycleState.EVALUATED,
        date="2026-05-10T10:00:00Z",
        date_kind="created_at",
        provider="openai",
        model="gpt-5-nano",
        encoder_provider=None,
        encoder_model=None,
        decoder_provider="openai",
        decoder_model="gpt-5-nano",
        encoder_system_prompt=None,
        encoder_user_prompt=None,
        encoder_output=None,
        decoder_system_prompt=None,
        decoder_user_prompt="implement",
        decoder_output=output,
        is_partial=False,
        prompt_fidelity=PromptFidelity.EXACT_REQUEST,
        content_sha256=content_sha256(
            None, None, None, None, "implement", output
        ),
        extraction_warning=None,
    )


def _request() -> RequestRecord:
    return RequestRecord(
        generation_id="generation-1",
        source_record_id="project:pool:decoder-1",
        dataset=DatasetName.HUMAN_EVAL,
        task_record_id=_task_hash(),
        task_id="HumanEval/0",
        generation_mode=GenerationMode.DIRECT,
        prompt_fidelity=PromptFidelity.EXACT_REQUEST,
        encoder_prompt_fidelity=None,
        decoder_prompt_fidelity=PromptFidelity.EXACT_REQUEST,
        budget_mode=BudgetMode.NO_BUDGET,
        max_characters=None,
        source_project="project",
        source_pool="pool",
        source_table="pool_pool_samples",
        source_file="pool.jsonl.gz",
        source_line_number=1,
        source_sample_id="decoder-1",
        sample_idx=0,
        run_id=None,
        source_kind="task_prompt",
        source_attempt_count=2,
        finish_reason="stop",
        response_finish_reason="stop",
        encoder_source_record_id=None,
        encoder_config_id=None,
        decoder_config_id="openai/gpt-5-nano/minimal/v1",
        encoder_prompt_template_id=None,
        decoder_prompt_template_id="sentinel/official_prompt",
        encoder_provider=None,
        encoder_model=None,
        decoder_provider="openai",
        decoder_model="gpt-5-nano",
        encoder_reasoning_json=None,
        decoder_reasoning_json=canonical_json({"kind": "openai"}),
        encoder_temperature=None,
        decoder_temperature=0.0,
        encoder_top_p=None,
        decoder_top_p=1.0,
        encoder_max_tokens=None,
        decoder_max_tokens=None,
        key_values_json=canonical_json({}),
        request_json=canonical_json({"prompt": "implement"}),
        response_json=canonical_json({"text": "def f():\n    return 1\n"}),
        metadata_json=canonical_json({}),
        hints_json=canonical_json({"output_kind": "code_text"}),
        extraction_warning=None,
    )


def _encoder() -> EncoderArtifactRecord:
    return EncoderArtifactRecord(
        encoder_artifact_id="encoder-artifact-1",
        source_record_id="project:pool:encoder-1",
        dataset=DatasetName.HUMAN_EVAL,
        source_variant="canonical",
        task_record_id=_task_hash(),
        data_sample_id="human_eval/HumanEval/0/gt_solution@abc",
        task_id="HumanEval/0",
        stage=Stage.ENCODER,
        lifecycle_state=LifecycleState.ENCODER_ONLY,
        date="2026-05-10T09:59:00Z",
        date_kind="created_at",
        provider="openai",
        model="gpt-5-nano",
        encoder_system_prompt=None,
        encoder_user_prompt="describe",
        encoder_output="description",
        prompt_fidelity=PromptFidelity.EXACT_REQUEST,
        content_sha256=content_sha256(None, "describe", "description"),
        extraction_warning=None,
    )


def _task() -> TaskRecord:
    task_json = canonical_json({"prompt": "def f(): ...", "tests": []})
    task_hash = _task_hash()
    return TaskRecord(
        task_record_id=task_hash,
        dataset=DatasetName.HUMAN_EVAL,
        source_variant="canonical",
        task_id="HumanEval/0",
        language="python",
        dataset_id="evalplus/humanevalplus",
        split="test",
        data_sample_id="human_eval/HumanEval/0/gt_solution@abc",
        source_digest="abc",
        dataset_revision="v3",
        evaluator_kind="humaneval_plus",
        material_fidelity=TaskMaterialFidelity.PINNED_SNAPSHOT,
        task_json=task_json,
        content_sha256=task_hash,
    )


def _populate(writer: CorpusWriter) -> None:
    writer.add_source_record(
        _source("project:pool:decoder-1", stage=Stage.DECODER)
    )
    writer.add_source_record(
        _source("project:pool:encoder-1", stage=Stage.ENCODER)
    )
    writer.add_generation(_generation())
    writer.add_request(_request())
    writer.add_encoder_artifact(_encoder())
    writer.add_task(_task())


def _writer(tmp_path: Path, destination: Path) -> CorpusWriter:
    path, manifest = _source_manifest(tmp_path)
    return CorpusWriter(
        destination,
        source_manifest_path=path,
        source_manifest=manifest,
        adapter_name="synthetic",
        created_at="2026-08-08T12:00:00Z",
    )


def test_writes_separate_grains_with_declared_schemas_and_hashes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "corpus"
    writer = _writer(tmp_path, destination)
    _populate(writer)

    manifest = writer.publish()

    assert pl.read_parquet_schema(destination / "generations.parquet") == (
        GENERATION_SCHEMA
    )
    assert pl.read_parquet_schema(destination / "source_records.parquet") == (
        SOURCE_RECORD_SCHEMA
    )
    assert (
        pl.read_parquet_schema(destination / "encoder_artifacts.parquet")
        == ENCODER_ARTIFACT_SCHEMA
    )
    assert pl.read_parquet_schema(destination / "requests.parquet") == (
        REQUEST_SCHEMA
    )
    assert pl.read_parquet_schema(destination / "tasks.parquet") == TASK_SCHEMA
    assert manifest.generations.rows == manifest.requests.rows == 1
    assert manifest.source_records.rows == 2
    generation_columns = pl.read_parquet_schema(
        destination / "generations.parquet"
    ).names()
    assert "attempt_index" not in generation_columns
    assert "is_retry" not in generation_columns
    for summary in (
        manifest.generations,
        manifest.source_records,
        manifest.encoder_artifacts,
        manifest.requests,
        manifest.tasks,
    ):
        digest = hashlib.sha256((destination / summary.path).read_bytes())
        assert digest.hexdigest() == summary.sha256


def test_parquet_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = _writer(tmp_path, tmp_path / "first")
    _populate(first)
    first_manifest = first.publish()
    second = _writer(tmp_path, tmp_path / "second")
    _populate(second)
    second_manifest = second.publish()

    assert (
        first_manifest.generations.sha256 == second_manifest.generations.sha256
    )
    assert first_manifest.source_records.sha256 == (
        second_manifest.source_records.sha256
    )
    assert first_manifest.requests.sha256 == second_manifest.requests.sha256


def _populate_duplicate_ids(writer: CorpusWriter) -> None:
    source = _source("project:pool:decoder-1", stage=Stage.DECODER)
    writer.add_source_record(source)
    writer.add_source_record(source)


def _populate_missing_request(writer: CorpusWriter) -> None:
    writer.add_source_record(
        _source("project:pool:decoder-1", stage=Stage.DECODER)
    )
    writer.add_generation(_generation())


def _populate_mismatched_request(writer: CorpusWriter) -> None:
    writer.add_source_record(
        _source("project:pool:decoder-1", stage=Stage.DECODER)
    )
    writer.add_generation(_generation())
    writer.add_request(
        _request().model_copy(update={"task_id": "HumanEval/1"})
    )
    writer.add_task(_task())


def _populate_mismatched_task(writer: CorpusWriter) -> None:
    writer.add_source_record(
        _source("project:pool:decoder-1", stage=Stage.DECODER)
    )
    writer.add_generation(_generation())
    writer.add_request(_request())
    writer.add_task(_task().model_copy(update={"task_id": "HumanEval/1"}))


@pytest.mark.parametrize(
    ("populate", "expected_match"),
    [
        (_populate_duplicate_ids, "duplicate IDs"),
        (
            _populate_missing_request,
            "request provenance record validation failed",
        ),
        (_populate_mismatched_request, "mismatched_request_provenance=1"),
        (_populate_mismatched_task, "mismatched_generation_tasks=1"),
    ],
)
def test_publish_validation_fails_without_publishing(
    tmp_path: Path,
    populate: Callable[[CorpusWriter], None],
    expected_match: str,
) -> None:
    destination = tmp_path / "corpus"
    writer = _writer(tmp_path, destination)
    populate(writer)

    with pytest.raises(ValueError, match=expected_match):
        writer.publish()
    assert not destination.exists()


def test_refuses_nonempty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "corpus"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty destination"):
        _writer(tmp_path, destination)
    assert marker.read_text(encoding="utf-8") == "keep"
