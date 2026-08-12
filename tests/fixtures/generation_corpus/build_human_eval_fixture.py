#!/usr/bin/env python3

"""Build the minimal HumanEval generation corpus fixture bundle."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from dr_code.generation_corpus.models import BuildManifest
from dr_code.generation_corpus.writer import (
    ENCODER_ARTIFACT_SCHEMA,
    GENERATION_SCHEMA,
    REQUEST_SCHEMA,
    SOURCE_RECORD_SCHEMA,
    TASK_SCHEMA,
    _schema_sha256,
)

_FIXTURE_ROOT = Path(__file__).resolve().parent / "human_eval"
_SOURCE_MANIFEST = "/fixture/source/manifest.json"
_CREATED_AT = "2026-08-10T00:00:00+00:00"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_table(
    path: Path, schema: pl.Schema, rows: list[dict[str, object]]
) -> None:
    frame = pl.DataFrame(rows, schema=schema)
    frame.write_parquet(path, compression="zstd")


def _artifact_summary(path: Path, schema: pl.Schema) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "rows": pl.read_parquet(path).height,
        "schema_sha256": _schema_sha256(schema),
    }


def build_fixture() -> Path:
    _FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    generations_rows = [
        {
            "generation_id": "direct",
            "source_record_id": "source-direct",
            "dataset": "human_eval",
            "source_variant": "primary",
            "task_record_id": "task-0",
            "data_sample_id": "HumanEval/0",
            "task_id": "HumanEval/0",
            "generation_mode": "direct",
            "stage": "decoder",
            "lifecycle_state": "pending_validation",
            "date": _CREATED_AT,
            "date_kind": "created_at",
            "provider": "openai",
            "model": "model-a",
            "encoder_provider": None,
            "encoder_model": None,
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_system_prompt": None,
            "encoder_user_prompt": None,
            "encoder_output": None,
            "decoder_system_prompt": None,
            "decoder_user_prompt": None,
            "decoder_output": "def f(x):\n    return x\n",
            "is_partial": False,
            "prompt_fidelity": "exact_request",
            "content_sha256": "a" * 64,
            "extraction_warning": None,
        },
        {
            "generation_id": "enc-no-budget",
            "source_record_id": "source-enc-no-budget",
            "dataset": "human_eval",
            "source_variant": "primary",
            "task_record_id": "task-0",
            "data_sample_id": "HumanEval/0",
            "task_id": "HumanEval/0",
            "generation_mode": "enc_dec",
            "stage": "decoder",
            "lifecycle_state": "pending_validation",
            "date": _CREATED_AT,
            "date_kind": "created_at",
            "provider": "openai",
            "model": "model-a",
            "encoder_provider": "openai",
            "encoder_model": "model-a",
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_system_prompt": None,
            "encoder_user_prompt": '{"code":"def f(): pass"}',
            "encoder_output": "description",
            "decoder_system_prompt": None,
            "decoder_user_prompt": None,
            "decoder_output": "def f(x):\n    return x\n",
            "is_partial": False,
            "prompt_fidelity": "exact_request",
            "content_sha256": "b" * 64,
            "extraction_warning": None,
        },
        {
            "generation_id": "enc-budget",
            "source_record_id": "source-enc-budget",
            "dataset": "human_eval",
            "source_variant": "primary",
            "task_record_id": "task-0",
            "data_sample_id": "HumanEval/0",
            "task_id": "HumanEval/0",
            "generation_mode": "enc_dec",
            "stage": "decoder",
            "lifecycle_state": "pending_validation",
            "date": _CREATED_AT,
            "date_kind": "created_at",
            "provider": "openai",
            "model": "model-a",
            "encoder_provider": "openai",
            "encoder_model": "model-a",
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_system_prompt": None,
            "encoder_user_prompt": (
                '{"code":"def f(): pass","max_characters":50}'
            ),
            "encoder_output": "short description",
            "decoder_system_prompt": None,
            "decoder_user_prompt": None,
            "decoder_output": "def f(x):\n    return x\n",
            "is_partial": False,
            "prompt_fidelity": "exact_request",
            "content_sha256": "c" * 64,
            "extraction_warning": None,
        },
        {
            "generation_id": "incomplete-encoder",
            "source_record_id": "source-incomplete",
            "dataset": "human_eval",
            "source_variant": "primary",
            "task_record_id": "task-0",
            "data_sample_id": "HumanEval/0",
            "task_id": "HumanEval/0",
            "generation_mode": "enc_dec",
            "stage": "decoder",
            "lifecycle_state": "pending_validation",
            "date": _CREATED_AT,
            "date_kind": "created_at",
            "provider": "openai",
            "model": "model-a",
            "encoder_provider": "openai",
            "encoder_model": "model-a",
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_system_prompt": None,
            "encoder_user_prompt": (
                '{"code":"def f(): pass","max_characters":50}'
            ),
            "encoder_output": None,
            "decoder_system_prompt": None,
            "decoder_user_prompt": None,
            "decoder_output": "def f(x):\n    return x\n",
            "is_partial": False,
            "prompt_fidelity": "exact_request",
            "content_sha256": "d" * 64,
            "extraction_warning": None,
        },
        {
            "generation_id": "blank",
            "source_record_id": "source-blank",
            "dataset": "human_eval",
            "source_variant": "primary",
            "task_record_id": "task-0",
            "data_sample_id": "HumanEval/0",
            "task_id": "HumanEval/0",
            "generation_mode": "direct",
            "stage": "decoder",
            "lifecycle_state": "pending_validation",
            "date": _CREATED_AT,
            "date_kind": "created_at",
            "provider": "openai",
            "model": "model-a",
            "encoder_provider": None,
            "encoder_model": None,
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_system_prompt": None,
            "encoder_user_prompt": None,
            "encoder_output": None,
            "decoder_system_prompt": None,
            "decoder_user_prompt": None,
            "decoder_output": "  ",
            "is_partial": False,
            "prompt_fidelity": "exact_request",
            "content_sha256": "e" * 64,
            "extraction_warning": None,
        },
        {
            "generation_id": "unresolved",
            "source_record_id": "source-unresolved",
            "dataset": "human_eval",
            "source_variant": "primary",
            "task_record_id": "task-0",
            "data_sample_id": "HumanEval/0",
            "task_id": "HumanEval/0",
            "generation_mode": "unresolved_encoder",
            "stage": "decoder",
            "lifecycle_state": "pending_validation",
            "date": _CREATED_AT,
            "date_kind": "created_at",
            "provider": "openai",
            "model": "model-a",
            "encoder_provider": None,
            "encoder_model": "model-a",
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_system_prompt": None,
            "encoder_user_prompt": None,
            "encoder_output": None,
            "decoder_system_prompt": None,
            "decoder_user_prompt": None,
            "decoder_output": "def f(x):\n    return x\n",
            "is_partial": False,
            "prompt_fidelity": "exact_request",
            "content_sha256": "f" * 64,
            "extraction_warning": "encoder_reference_unavailable",
        },
    ]
    requests_rows = [
        {
            "generation_id": row["generation_id"],
            "source_record_id": row["source_record_id"],
            "dataset": "human_eval",
            "task_record_id": "task-0",
            "task_id": "HumanEval/0",
            "generation_mode": row["generation_mode"],
            "prompt_fidelity": "exact_request",
            "encoder_prompt_fidelity": None,
            "decoder_prompt_fidelity": "exact_request",
            "budget_mode": (
                "budget"
                if row["generation_id"] == "enc-budget"
                else "no_budget"
                if row["generation_mode"] in {"direct", "enc_dec"}
                else "unresolved"
            ),
            "max_characters": 50
            if row["generation_id"] == "enc-budget"
            else None,
            "source_project": "fixture",
            "source_pool": "decoder_pool",
            "source_table": "decoder",
            "source_file": "decoder.ndjson.gz",
            "source_line_number": index + 1,
            "source_sample_id": row["generation_id"],
            "sample_idx": index,
            "run_id": "run-1",
            "source_kind": "legacy_dbos_generation_attempt",
            "source_attempt_count": 1,
            "finish_reason": "stop",
            "response_finish_reason": "stop",
            "encoder_source_record_id": None,
            "encoder_config_id": None,
            "decoder_config_id": "dec-config",
            "encoder_prompt_template_id": None,
            "decoder_prompt_template_id": "dec-template",
            "encoder_provider": row["encoder_provider"],
            "encoder_model": row["encoder_model"],
            "decoder_provider": "openai",
            "decoder_model": "model-a",
            "encoder_reasoning_json": None,
            "decoder_reasoning_json": None,
            "encoder_temperature": None,
            "decoder_temperature": 0.0,
            "encoder_top_p": None,
            "decoder_top_p": 1.0,
            "encoder_max_tokens": None,
            "decoder_max_tokens": 512,
            "key_values_json": "{}",
            "request_json": "{}",
            "response_json": "{}",
            "metadata_json": "{}",
            "hints_json": "{}",
            "extraction_warning": row["extraction_warning"],
        }
        for index, row in enumerate(generations_rows)
    ]

    generations_path = _FIXTURE_ROOT / "generations.parquet"
    requests_path = _FIXTURE_ROOT / "requests.parquet"
    source_records_path = _FIXTURE_ROOT / "source_records.parquet"
    encoder_artifacts_path = _FIXTURE_ROOT / "encoder_artifacts.parquet"
    tasks_path = _FIXTURE_ROOT / "tasks.parquet"

    _write_table(generations_path, GENERATION_SCHEMA, generations_rows)
    _write_table(requests_path, REQUEST_SCHEMA, requests_rows)
    _write_table(source_records_path, SOURCE_RECORD_SCHEMA, [])
    _write_table(encoder_artifacts_path, ENCODER_ARTIFACT_SCHEMA, [])
    _write_table(
        tasks_path,
        TASK_SCHEMA,
        [
            {
                "task_record_id": "task-0",
                "dataset": "human_eval",
                "source_variant": "primary",
                "task_id": "HumanEval/0",
                "language": "python",
                "dataset_id": "HumanEval",
                "split": "test",
                "data_sample_id": "HumanEval/0",
                "source_digest": "digest",
                "dataset_revision": None,
                "evaluator_kind": "humaneval",
                "material_fidelity": "pinned_snapshot",
                "task_json": "{}",
                "content_sha256": "0" * 64,
            }
        ],
    )

    manifest = BuildManifest(
        adapter_name="fixture_human_eval",
        adapter_version=1,
        created_at=datetime.now(tz=UTC).isoformat(),
        source_manifest=_SOURCE_MANIFEST,
        source_manifest_sha256="1" * 64,
        source_dump_created_at=_CREATED_AT,
        source_dump_pool_count=1,
        generations=_artifact_summary(generations_path, GENERATION_SCHEMA),
        source_records=_artifact_summary(
            source_records_path, SOURCE_RECORD_SCHEMA
        ),
        encoder_artifacts=_artifact_summary(
            encoder_artifacts_path,
            ENCODER_ARTIFACT_SCHEMA,
        ),
        requests=_artifact_summary(requests_path, REQUEST_SCHEMA),
        tasks=_artifact_summary(tasks_path, TASK_SCHEMA),
    )
    manifest_path = _FIXTURE_ROOT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return _FIXTURE_ROOT


if __name__ == "__main__":
    print(build_fixture())
