from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO

import polars as pl

from dr_code.core.models import FrozenModel
from dr_code.generation_corpus.models import (
    ArtifactSummary,
    BuildManifest,
    EncoderArtifactRecord,
    GenerationMode,
    GenerationRecord,
    RequestRecord,
    SourceManifest,
    SourceRecord,
    Stage,
    TaskRecord,
)
from dr_code.generation_corpus.pool_dump import canonical_json, content_sha256

_MANIFEST_NAME: Final = "manifest.json"
_TABLE_FILES: Final = {
    "generations": "generations.parquet",
    "source_records": "source_records.parquet",
    "encoder_artifacts": "encoder_artifacts.parquet",
    "requests": "requests.parquet",
    "tasks": "tasks.parquet",
}

GENERATION_SCHEMA: Final = pl.Schema(
    {
        "generation_id": pl.String,
        "source_record_id": pl.String,
        "dataset": pl.String,
        "source_variant": pl.String,
        "task_record_id": pl.String,
        "data_sample_id": pl.String,
        "task_id": pl.String,
        "generation_mode": pl.String,
        "stage": pl.String,
        "lifecycle_state": pl.String,
        "date": pl.String,
        "date_kind": pl.String,
        "provider": pl.String,
        "model": pl.String,
        "encoder_provider": pl.String,
        "encoder_model": pl.String,
        "decoder_provider": pl.String,
        "decoder_model": pl.String,
        "encoder_system_prompt": pl.String,
        "encoder_user_prompt": pl.String,
        "encoder_output": pl.String,
        "decoder_system_prompt": pl.String,
        "decoder_user_prompt": pl.String,
        "decoder_output": pl.String,
        "is_partial": pl.Boolean,
        "prompt_fidelity": pl.String,
        "content_sha256": pl.String,
        "extraction_warning": pl.String,
    }
)

SOURCE_RECORD_SCHEMA: Final = pl.Schema(
    {
        "source_record_id": pl.String,
        "dataset": pl.String,
        "source_variant": pl.String,
        "data_sample_id": pl.String,
        "task_id": pl.String,
        "source_project": pl.String,
        "source_pool": pl.String,
        "source_table": pl.String,
        "source_file": pl.String,
        "source_line_number": pl.Int64,
        "source_sample_id": pl.String,
        "sample_idx": pl.Int64,
        "run_id": pl.String,
        "stage": pl.String,
        "lifecycle_state": pl.String,
        "date": pl.String,
        "date_kind": pl.String,
        "status": pl.String,
        "attempt_count": pl.Int64,
        "finish_reason": pl.String,
        "response_finish_reason": pl.String,
        "has_nonblank_output": pl.Boolean,
        "output_text": pl.String,
        "key_values_json": pl.String,
        "request_json": pl.String,
        "response_json": pl.String,
        "metadata_json": pl.String,
        "hints_json": pl.String,
    }
)

ENCODER_ARTIFACT_SCHEMA: Final = pl.Schema(
    {
        "encoder_artifact_id": pl.String,
        "source_record_id": pl.String,
        "dataset": pl.String,
        "source_variant": pl.String,
        "task_record_id": pl.String,
        "data_sample_id": pl.String,
        "task_id": pl.String,
        "stage": pl.String,
        "lifecycle_state": pl.String,
        "date": pl.String,
        "date_kind": pl.String,
        "provider": pl.String,
        "model": pl.String,
        "encoder_system_prompt": pl.String,
        "encoder_user_prompt": pl.String,
        "encoder_output": pl.String,
        "prompt_fidelity": pl.String,
        "content_sha256": pl.String,
        "extraction_warning": pl.String,
    }
)

REQUEST_SCHEMA: Final = pl.Schema(
    {
        "generation_id": pl.String,
        "source_record_id": pl.String,
        "dataset": pl.String,
        "task_record_id": pl.String,
        "task_id": pl.String,
        "generation_mode": pl.String,
        "prompt_fidelity": pl.String,
        "encoder_prompt_fidelity": pl.String,
        "decoder_prompt_fidelity": pl.String,
        "budget_mode": pl.String,
        "max_characters": pl.Int64,
        "source_project": pl.String,
        "source_pool": pl.String,
        "source_table": pl.String,
        "source_file": pl.String,
        "source_line_number": pl.Int64,
        "source_sample_id": pl.String,
        "sample_idx": pl.Int64,
        "run_id": pl.String,
        "source_kind": pl.String,
        "source_attempt_count": pl.Int64,
        "finish_reason": pl.String,
        "response_finish_reason": pl.String,
        "encoder_source_record_id": pl.String,
        "encoder_config_id": pl.String,
        "decoder_config_id": pl.String,
        "encoder_prompt_template_id": pl.String,
        "decoder_prompt_template_id": pl.String,
        "encoder_provider": pl.String,
        "encoder_model": pl.String,
        "decoder_provider": pl.String,
        "decoder_model": pl.String,
        "encoder_reasoning_json": pl.String,
        "decoder_reasoning_json": pl.String,
        "encoder_temperature": pl.Float64,
        "decoder_temperature": pl.Float64,
        "encoder_top_p": pl.Float64,
        "decoder_top_p": pl.Float64,
        "encoder_max_tokens": pl.Int64,
        "decoder_max_tokens": pl.Int64,
        "key_values_json": pl.String,
        "request_json": pl.String,
        "response_json": pl.String,
        "metadata_json": pl.String,
        "hints_json": pl.String,
        "extraction_warning": pl.String,
    }
)

TASK_SCHEMA: Final = pl.Schema(
    {
        "task_record_id": pl.String,
        "dataset": pl.String,
        "source_variant": pl.String,
        "task_id": pl.String,
        "language": pl.String,
        "dataset_id": pl.String,
        "split": pl.String,
        "data_sample_id": pl.String,
        "source_digest": pl.String,
        "dataset_revision": pl.String,
        "evaluator_kind": pl.String,
        "material_fidelity": pl.String,
        "task_json": pl.String,
        "content_sha256": pl.String,
    }
)

_SCHEMAS: Final = {
    "generations": GENERATION_SCHEMA,
    "source_records": SOURCE_RECORD_SCHEMA,
    "encoder_artifacts": ENCODER_ARTIFACT_SCHEMA,
    "requests": REQUEST_SCHEMA,
    "tasks": TASK_SCHEMA,
}
_ID_COLUMNS: Final = {
    "generations": "generation_id",
    "source_records": "source_record_id",
    "encoder_artifacts": "encoder_artifact_id",
    "requests": "generation_id",
    "tasks": "task_record_id",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(schema: pl.Schema) -> str:
    payload = [[name, str(dtype)] for name, dtype in schema.items()]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusPopulation:
    generations: int
    source_records: int
    encoder_artifacts: int
    requests: int
    tasks: int

    def __post_init__(self) -> None:
        if any(
            count < 0
            for count in (
                self.generations,
                self.source_records,
                self.encoder_artifacts,
                self.requests,
                self.tasks,
            )
        ):
            raise ValueError("corpus population counts must be nonnegative")


class CorpusWriter:
    """Stream corpus records to validated Parquet and publish one directory."""

    def __init__(
        self,
        destination: Path,
        *,
        source_manifest_path: Path,
        source_manifest: SourceManifest,
        adapter_name: str,
        adapter_version: int = 1,
        created_at: str | None = None,
    ) -> None:
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(
                f"refusing to write into non-empty destination {destination}"
            )
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._destination = destination
        self._source_manifest_path = source_manifest_path
        self._source_manifest = source_manifest
        self._adapter_name = adapter_name
        self._adapter_version = adapter_version
        self._created_at = created_at or datetime.now(UTC).isoformat()
        self._temporary = tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-", dir=destination.parent
        )
        self._temporary_path = Path(self._temporary.name)
        self._bundle_path = self._temporary_path / "bundle"
        self._bundle_path.mkdir()
        self._handles: dict[str, TextIO] = {
            name: (self._temporary_path / f"{name}.ndjson").open(
                "w", encoding="utf-8"
            )
            for name in _SCHEMAS
        }
        self._counts = dict.fromkeys(_SCHEMAS, 0)
        self._closed = False
        self._published = False

    def __enter__(self) -> CorpusWriter:
        return self

    def __exit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        if exc_type is None:
            self.publish()
        else:
            self.abort()

    def add_generation(self, record: GenerationRecord) -> None:
        self._add("generations", record)

    def add_source_record(self, record: SourceRecord) -> None:
        self._add("source_records", record)

    def add_encoder_artifact(self, record: EncoderArtifactRecord) -> None:
        self._add("encoder_artifacts", record)

    def add_request(self, record: RequestRecord) -> None:
        self._add("requests", record)

    def add_task(self, record: TaskRecord) -> None:
        self._add("tasks", record)

    @property
    def population(self) -> CorpusPopulation:
        return CorpusPopulation(
            generations=self._counts["generations"],
            source_records=self._counts["source_records"],
            encoder_artifacts=self._counts["encoder_artifacts"],
            requests=self._counts["requests"],
            tasks=self._counts["tasks"],
        )

    def _add(self, table: str, record: FrozenModel) -> None:
        if self._closed:
            raise RuntimeError("corpus writer is closed")
        self._validate_content_hash(record)
        payload = record.model_dump(mode="json")
        self._handles[table].write(canonical_json(payload))
        self._handles[table].write("\n")
        self._counts[table] += 1

    @staticmethod
    def _validate_content_hash(record: FrozenModel) -> None:
        expected: str | None = None
        actual: str | None = None
        if isinstance(record, GenerationRecord):
            actual = record.content_sha256
            expected = content_sha256(
                record.encoder_system_prompt,
                record.encoder_user_prompt,
                record.encoder_output,
                record.decoder_system_prompt,
                record.decoder_user_prompt,
                record.decoder_output,
            )
        elif isinstance(record, EncoderArtifactRecord):
            actual = record.content_sha256
            expected = content_sha256(
                record.encoder_system_prompt,
                record.encoder_user_prompt,
                record.encoder_output,
            )
        elif isinstance(record, TaskRecord):
            actual = record.content_sha256
            expected = content_sha256(json.loads(record.task_json))
            if record.task_record_id != expected:
                raise ValueError(
                    "task_record_id must equal the task material content hash"
                )
        if expected is not None and actual != expected:
            raise ValueError(
                "record content_sha256 does not match its content"
            )

    def close(self) -> BuildManifest:
        return self.publish()

    def publish(self) -> BuildManifest:
        if self._published:
            raise RuntimeError("corpus writer has already published")
        if self._closed:
            raise RuntimeError("corpus writer is closed")
        self._close_handles()
        try:
            for table, schema in _SCHEMAS.items():
                self._ndjson_to_parquet(
                    self._temporary_path / f"{table}.ndjson",
                    self._bundle_path / _TABLE_FILES[table],
                    schema,
                )
            self._validate_tables()
            manifest = self._build_manifest()
            manifest_path = self._bundle_path / _MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(
                    manifest.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if self._destination.exists():
                self._destination.rmdir()
            os.replace(self._bundle_path, self._destination)
        except BaseException:
            self._closed = True
            self._temporary.cleanup()
            raise
        self._published = True
        self._closed = True
        self._temporary.cleanup()
        return manifest

    def abort(self) -> None:
        if not self._closed:
            self._close_handles()
            self._temporary.cleanup()

    def _close_handles(self) -> None:
        if self._closed:
            return
        for handle in self._handles.values():
            handle.close()
        self._closed = True

    @staticmethod
    def _ndjson_to_parquet(
        ndjson: Path, parquet: Path, schema: pl.Schema
    ) -> None:
        if ndjson.stat().st_size == 0:
            pl.DataFrame(schema=schema).write_parquet(
                parquet,
                compression="zstd",
                compression_level=3,
                statistics=False,
            )
            return
        pl.scan_ndjson(ndjson, schema=schema).sink_parquet(
            parquet,
            compression="zstd",
            compression_level=3,
            statistics=False,
            maintain_order=True,
        )

    def _validate_tables(self) -> None:
        paths = {
            table: self._bundle_path / file_name
            for table, file_name in _TABLE_FILES.items()
        }
        for table, path in paths.items():
            expected_schema = _SCHEMAS[table]
            actual_schema = pl.read_parquet_schema(path)
            if actual_schema != expected_schema:
                raise ValueError(
                    f"schema mismatch for {table}: "
                    f"expected={expected_schema}, actual={actual_schema}"
                )
            checks = (
                pl.scan_parquet(path)
                .select(
                    pl.len().alias("rows"),
                    pl.col(_ID_COLUMNS[table]).n_unique().alias("unique_ids"),
                )
                .collect()
                .row(0, named=True)
            )
            if checks["rows"] != self._counts[table]:
                raise ValueError(
                    f"row count mismatch for {table}: "
                    f"declared={self._counts[table]}, actual={checks['rows']}"
                )
            if checks["unique_ids"] != checks["rows"]:
                raise ValueError(f"duplicate IDs in {table}")

        generations = pl.scan_parquet(paths["generations"]).select(
            "generation_id",
            "source_record_id",
            "dataset",
            "source_variant",
            "task_record_id",
            "data_sample_id",
            "task_id",
            "generation_mode",
            "lifecycle_state",
            "stage",
            "encoder_model",
            "encoder_output",
            "decoder_output",
            "prompt_fidelity",
        )
        requests = pl.scan_parquet(paths["requests"]).select(
            "generation_id",
            "source_record_id",
            "dataset",
            "task_record_id",
            "task_id",
            "generation_mode",
            "prompt_fidelity",
            "encoder_source_record_id",
            "source_project",
            "source_pool",
            "source_table",
            "source_file",
            "source_line_number",
            "source_sample_id",
            "sample_idx",
            "run_id",
            "source_attempt_count",
            "finish_reason",
            "response_finish_reason",
            "key_values_json",
            "request_json",
            "response_json",
            "metadata_json",
            "hints_json",
        )
        missing_requests = generations.join(
            requests.select("generation_id"), on="generation_id", how="anti"
        ).select(pl.len())
        extra_requests = requests.join(
            generations.select("generation_id"), on="generation_id", how="anti"
        ).select(pl.len())
        generation_requests = generations.join(
            requests,
            on="generation_id",
            how="inner",
            suffix="_request",
            validate="1:1",
        )
        mismatched_request_provenance = generation_requests.filter(
            (pl.col("source_record_id") != pl.col("source_record_id_request"))
            | (pl.col("dataset") != pl.col("dataset_request"))
            | (pl.col("task_record_id") != pl.col("task_record_id_request"))
            | (pl.col("task_id") != pl.col("task_id_request"))
            | (pl.col("generation_mode") != pl.col("generation_mode_request"))
            | (pl.col("prompt_fidelity") != pl.col("prompt_fidelity_request"))
        )
        sources = pl.scan_parquet(paths["source_records"]).select(
            "source_record_id",
            "dataset",
            "source_variant",
            "data_sample_id",
            "task_id",
            "source_project",
            "source_pool",
            "source_table",
            "source_file",
            "source_line_number",
            "source_sample_id",
            "sample_idx",
            "run_id",
            "stage",
            "lifecycle_state",
            "attempt_count",
            "finish_reason",
            "response_finish_reason",
            "output_text",
            "key_values_json",
            "request_json",
            "response_json",
            "metadata_json",
            "hints_json",
        )
        source_ids = sources.select("source_record_id")
        missing_generation_sources = generations.select(
            "source_record_id"
        ).join(source_ids, on="source_record_id", how="anti")
        generation_sources = generations.join(
            sources,
            on="source_record_id",
            how="inner",
            suffix="_source",
            validate="1:1",
        )
        mismatched_generation_sources = generation_sources.filter(
            (pl.col("dataset") != pl.col("dataset_source"))
            | (pl.col("source_variant") != pl.col("source_variant_source"))
            | ~pl.col("data_sample_id").eq_missing(
                pl.col("data_sample_id_source")
            )
            | (pl.col("task_id") != pl.col("task_id_source"))
            | (pl.col("lifecycle_state") != pl.col("lifecycle_state_source"))
            | (pl.col("stage_source") != Stage.DECODER.value)
            | (pl.col("decoder_output") != pl.col("output_text"))
        )
        request_sources = requests.join(
            sources,
            on="source_record_id",
            how="inner",
            suffix="_source",
            validate="1:1",
        )
        mismatched_request_sources = request_sources.filter(
            (pl.col("dataset") != pl.col("dataset_source"))
            | (pl.col("task_id") != pl.col("task_id_source"))
            | (pl.col("source_project") != pl.col("source_project_source"))
            | (pl.col("source_pool") != pl.col("source_pool_source"))
            | (pl.col("source_table") != pl.col("source_table_source"))
            | (pl.col("source_file") != pl.col("source_file_source"))
            | (
                pl.col("source_line_number")
                != pl.col("source_line_number_source")
            )
            | (pl.col("source_sample_id") != pl.col("source_sample_id_source"))
            | ~pl.col("sample_idx").eq_missing(pl.col("sample_idx_source"))
            | ~pl.col("run_id").eq_missing(pl.col("run_id_source"))
            | (pl.col("source_attempt_count") != pl.col("attempt_count"))
            | ~pl.col("finish_reason").eq_missing(
                pl.col("finish_reason_source")
            )
            | ~pl.col("response_finish_reason").eq_missing(
                pl.col("response_finish_reason_source")
            )
            | (pl.col("key_values_json") != pl.col("key_values_json_source"))
            | (pl.col("request_json") != pl.col("request_json_source"))
            | (pl.col("response_json") != pl.col("response_json_source"))
            | (pl.col("metadata_json") != pl.col("metadata_json_source"))
            | (pl.col("hints_json") != pl.col("hints_json_source"))
        )
        encoder_artifacts = pl.scan_parquet(paths["encoder_artifacts"]).select(
            "source_record_id",
            "dataset",
            "source_variant",
            "task_record_id",
            "data_sample_id",
            "task_id",
            "stage",
            "lifecycle_state",
            "encoder_output",
        )
        missing_encoder_sources = encoder_artifacts.select(
            "source_record_id"
        ).join(source_ids, on="source_record_id", how="anti")
        mismatched_encoder_sources = encoder_artifacts.join(
            sources,
            on="source_record_id",
            how="inner",
            suffix="_source",
            validate="1:1",
        ).filter(
            (pl.col("dataset") != pl.col("dataset_source"))
            | (pl.col("source_variant") != pl.col("source_variant_source"))
            | ~pl.col("data_sample_id").eq_missing(
                pl.col("data_sample_id_source")
            )
            | ~pl.col("task_id").eq_missing(pl.col("task_id_source"))
            | (pl.col("stage") != Stage.ENCODER.value)
            | (pl.col("stage_source") != Stage.ENCODER.value)
            | (pl.col("lifecycle_state") != pl.col("lifecycle_state_source"))
            | (pl.col("encoder_output") != pl.col("output_text"))
        )
        missing_request_encoder_sources = (
            requests.filter(pl.col("encoder_source_record_id").is_not_null())
            .select(
                pl.col("encoder_source_record_id").alias("source_record_id")
            )
            .join(source_ids, on="source_record_id", how="anti")
        )
        tasks = pl.scan_parquet(paths["tasks"]).select(
            "task_record_id",
            "dataset",
            "task_id",
            "data_sample_id",
        )
        task_ids = tasks.select("task_record_id")
        missing_generation_tasks = generations.select("task_record_id").join(
            task_ids, on="task_record_id", how="anti"
        )
        mismatched_generation_tasks = generations.join(
            tasks,
            on="task_record_id",
            how="inner",
            suffix="_task",
            validate="m:1",
        ).filter(
            (pl.col("dataset") != pl.col("dataset_task"))
            | (pl.col("task_id") != pl.col("task_id_task"))
            | (
                pl.col("data_sample_id_task").is_not_null()
                & ~pl.col("data_sample_id").eq_missing(
                    pl.col("data_sample_id_task")
                )
            )
        )
        missing_encoder_tasks = (
            encoder_artifacts.filter(pl.col("task_record_id").is_not_null())
            .select("task_record_id")
            .join(task_ids, on="task_record_id", how="anti")
        )
        mismatched_encoder_tasks = (
            encoder_artifacts.filter(pl.col("task_record_id").is_not_null())
            .join(
                tasks,
                on="task_record_id",
                how="inner",
                suffix="_task",
                validate="m:1",
            )
            .filter(
                (pl.col("dataset") != pl.col("dataset_task"))
                | (pl.col("task_id") != pl.col("task_id_task"))
                | (
                    pl.col("data_sample_id_task").is_not_null()
                    & ~pl.col("data_sample_id").eq_missing(
                        pl.col("data_sample_id_task")
                    )
                )
            )
        )
        encoder_source_records = sources.select(
            pl.col("source_record_id").alias("encoder_source_record_id"),
            pl.col("dataset").alias("encoder_source_dataset"),
            pl.col("source_variant").alias("encoder_source_variant"),
            pl.col("data_sample_id").alias("encoder_source_data_sample_id"),
            pl.col("task_id").alias("encoder_source_task_id"),
            pl.col("stage").alias("encoder_source_stage"),
            pl.col("output_text").alias("encoder_source_output"),
        )
        mismatched_encoder_lineage = (
            generation_requests.filter(
                pl.col("encoder_source_record_id").is_not_null()
            )
            .join(
                encoder_source_records,
                on="encoder_source_record_id",
                how="inner",
                validate="m:1",
            )
            .filter(
                (
                    pl.col("generation_mode")
                    != GenerationMode.ENCODER_DECODER.value
                )
                | (pl.col("encoder_source_stage") != Stage.ENCODER.value)
                | (pl.col("dataset") != pl.col("encoder_source_dataset"))
                | (
                    pl.col("source_variant")
                    != pl.col("encoder_source_variant")
                )
                | ~pl.col("data_sample_id").eq_missing(
                    pl.col("encoder_source_data_sample_id")
                )
                | (pl.col("task_id") != pl.col("encoder_source_task_id"))
                | (pl.col("encoder_output") != pl.col("encoder_source_output"))
            )
        )
        incomplete_enc_dec = generations.filter(
            (pl.col("generation_mode") == GenerationMode.ENCODER_DECODER.value)
            & (
                pl.col("encoder_model").is_null()
                | pl.col("encoder_output").is_null()
            )
        )
        request_provenance_checks = pl.collect_all(
            [
                missing_requests,
                extra_requests,
                mismatched_request_provenance.select(pl.len()),
                missing_generation_sources.select(pl.len()),
                mismatched_generation_sources.select(pl.len()),
                mismatched_request_sources.select(pl.len()),
                missing_encoder_sources.select(pl.len()),
                mismatched_encoder_sources.select(pl.len()),
                missing_request_encoder_sources.select(pl.len()),
                missing_generation_tasks.select(pl.len()),
                mismatched_generation_tasks.select(pl.len()),
                missing_encoder_tasks.select(pl.len()),
                mismatched_encoder_tasks.select(pl.len()),
                mismatched_encoder_lineage.select(pl.len()),
                incomplete_enc_dec.select(pl.len()),
            ]
        )
        values = [frame.item() for frame in request_provenance_checks]
        if any(values):
            raise ValueError(
                "corpus request provenance record validation failed: "
                f"missing_requests={values[0]}, extra_requests={values[1]}, "
                f"mismatched_request_provenance={values[2]}, "
                f"missing_generation_sources={values[3]}, "
                f"mismatched_generation_sources={values[4]}, "
                f"mismatched_request_sources={values[5]}, "
                f"missing_encoder_sources={values[6]}, "
                f"mismatched_encoder_sources={values[7]}, "
                f"missing_request_encoder_sources={values[8]}, "
                f"missing_generation_tasks={values[9]}, "
                f"mismatched_generation_tasks={values[10]}, "
                f"missing_encoder_tasks={values[11]}, "
                f"mismatched_encoder_tasks={values[12]}, "
                f"mismatched_encoder_lineage={values[13]}, "
                f"incomplete_enc_dec={values[14]}"
            )

    def _build_manifest(self) -> BuildManifest:
        summaries: dict[str, ArtifactSummary] = {}
        for table, file_name in _TABLE_FILES.items():
            path = self._bundle_path / file_name
            summaries[table] = ArtifactSummary(
                path=file_name,
                sha256=_sha256_file(path),
                rows=self._counts[table],
                schema_sha256=_schema_sha256(_SCHEMAS[table]),
            )
        return BuildManifest(
            adapter_name=self._adapter_name,
            adapter_version=self._adapter_version,
            created_at=self._created_at,
            source_manifest=str(self._source_manifest_path),
            source_manifest_sha256=_sha256_file(self._source_manifest_path),
            source_dump_created_at=self._source_manifest.created_at,
            source_dump_pool_count=len(self._source_manifest.pools),
            generations=summaries["generations"],
            source_records=summaries["source_records"],
            encoder_artifacts=summaries["encoder_artifacts"],
            requests=summaries["requests"],
            tasks=summaries["tasks"],
        )


__all__ = [
    "CorpusPopulation",
    "CorpusWriter",
    "ENCODER_ARTIFACT_SCHEMA",
    "GENERATION_SCHEMA",
    "REQUEST_SCHEMA",
    "SOURCE_RECORD_SCHEMA",
    "TASK_SCHEMA",
]
