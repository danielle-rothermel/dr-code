from __future__ import annotations

from datetime import datetime
from enum import UNIQUE, StrEnum, verify
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, JsonValue, field_validator

from dr_code.core.models import FrozenModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]


def _validate_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value


TimestampString = Annotated[str, AfterValidator(_validate_timestamp)]


# These enums are persisted contracts. Build payloads from named members, never
# by iterating over the enum and depending on declaration order.
@verify(UNIQUE)
class DatasetName(StrEnum):
    HUMAN_EVAL = "human_eval"
    MBPP_PRO = "mbpp_pro"
    HUMANEVAL_PRO = "humaneval_pro"
    CLASS_EVAL = "class_eval"
    BIGCODEBENCH_LITE_PRO = "bigcodebench_lite_pro"
    NL_LATENTS = "nl_latents"


@verify(UNIQUE)
class GenerationMode(StrEnum):
    DIRECT = "direct"
    ENCODER_DECODER = "enc_dec"
    UNRESOLVED_ENCODER = "unresolved_encoder"


@verify(UNIQUE)
class BudgetMode(StrEnum):
    NO_BUDGET = "no_budget"
    BUDGET = "budget"
    UNRESOLVED = "unresolved"


@verify(UNIQUE)
class Stage(StrEnum):
    ENCODER = "encoder"
    DECODER = "decoder"


@verify(UNIQUE)
class PromptFidelity(StrEnum):
    EXACT_REQUEST = "exact_request"
    RECOVERED_TASK = "recovered_task"
    UNAVAILABLE = "unavailable"


@verify(UNIQUE)
class LifecycleState(StrEnum):
    EVALUATED = "evaluated"
    PENDING_VALIDATION = "pending_validation"
    ENCODER_ONLY = "encoder_only"
    PRE_ENCODER_FAILURE = "pre_encoder_failure"
    FAILED = "failed"
    SEEDED = "seeded"


@verify(UNIQUE)
class TaskMaterialFidelity(StrEnum):
    PINNED_SNAPSHOT = "pinned_snapshot"
    PERSISTED = "persisted"
    UNAVAILABLE = "unavailable"


@verify(UNIQUE)
class DumpOutputKind(StrEnum):
    CODE_TEXT = "code_text"
    DECODED_CODE = "decoded_code"
    NOT_CODE = "not_code"


@verify(UNIQUE)
class DecoderDescriptionSource(StrEnum):
    METADATA_SOURCE_TEXT = "metadata.source_text"
    SOURCE_SAMPLE_PAYLOAD_TEXT = "metadata.source_sample_payload.text"
    REQUEST_PROMPT = "request.prompt"
    HUMANEVAL_CACHE_PROMPT = "humaneval_cache.prompt"
    MISSING = "missing"


class PoolSchemaColumn(FrozenModel):
    name: NonEmptyString
    type: NonEmptyString


class PoolSchema(FrozenModel):
    name: NonEmptyString
    key_columns: tuple[PoolSchemaColumn, ...]


class PoolManifestEntry(FrozenModel):
    project_name: NonEmptyString
    pool_name: NonEmptyString
    table_name: NonEmptyString
    file_name: NonEmptyString
    row_count: Annotated[int, Field(ge=0)]
    dumped_row_count: Annotated[int, Field(ge=0)]
    pool_schema_json: PoolSchema
    original_status: NonEmptyString
    temporarily_started: bool


class SourceManifest(FrozenModel):
    version: Literal[1]
    created_at: TimestampString
    output_dir: NonEmptyString
    pools: tuple[PoolManifestEntry, ...]


class DumpHints(FrozenModel):
    human_eval_task_id: str | None
    human_eval_pro_task_id: str | None
    output_kind: DumpOutputKind
    output_json_path: str | None
    decoder_input_description_source: DecoderDescriptionSource


class DumpedPoolRow(FrozenModel):
    project_name: NonEmptyString
    pool_name: NonEmptyString
    table_name: NonEmptyString
    sample_id: NonEmptyString
    key_values: dict[str, JsonValue]
    sample_idx: int | None
    run_id: str | None
    request_json: dict[str, JsonValue]
    response_json: dict[str, JsonValue] | None
    finish_reason: str | None
    attempt_count: Annotated[int, Field(ge=0)]
    metadata_json: dict[str, JsonValue]
    created_at: TimestampString | None
    hints: DumpHints


class GenerationRecord(FrozenModel):
    generation_id: NonEmptyString
    source_record_id: NonEmptyString
    dataset: DatasetName
    source_variant: NonEmptyString
    task_record_id: NonEmptyString
    data_sample_id: str | None
    task_id: NonEmptyString
    generation_mode: GenerationMode
    stage: Stage
    lifecycle_state: LifecycleState
    date: TimestampString | None
    date_kind: NonEmptyString
    provider: str | None
    model: str | None
    encoder_provider: str | None
    encoder_model: str | None
    decoder_provider: str | None
    decoder_model: str | None
    encoder_system_prompt: str | None
    encoder_user_prompt: str | None
    encoder_output: str | None
    decoder_system_prompt: str | None
    decoder_user_prompt: str | None
    decoder_output: NonEmptyString
    is_partial: bool
    prompt_fidelity: PromptFidelity
    content_sha256: Sha256
    extraction_warning: str | None

    @field_validator("decoder_output")
    @classmethod
    def _require_nonblank_decoder_output(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("decoder_output must be nonblank")
        return value

    @field_validator("stage")
    @classmethod
    def _require_decoder_stage(cls, value: Stage) -> Stage:
        if value is not Stage.DECODER:
            raise ValueError("canonical generations must have decoder stage")
        return value


class SourceRecord(FrozenModel):
    source_record_id: NonEmptyString
    dataset: DatasetName
    source_variant: NonEmptyString
    data_sample_id: str | None
    task_id: str | None
    source_project: NonEmptyString
    source_pool: NonEmptyString
    source_table: NonEmptyString
    source_file: NonEmptyString
    source_line_number: Annotated[int, Field(ge=1)]
    source_sample_id: NonEmptyString
    sample_idx: int | None
    run_id: str | None
    stage: Stage
    lifecycle_state: LifecycleState
    date: TimestampString | None
    date_kind: NonEmptyString
    status: str | None
    attempt_count: Annotated[int, Field(ge=0)]
    finish_reason: str | None
    response_finish_reason: str | None
    has_nonblank_output: bool
    output_text: str | None
    key_values_json: NonEmptyString
    request_json: NonEmptyString
    response_json: NonEmptyString
    metadata_json: NonEmptyString
    hints_json: NonEmptyString


class EncoderArtifactRecord(FrozenModel):
    encoder_artifact_id: NonEmptyString
    source_record_id: NonEmptyString
    dataset: DatasetName
    source_variant: NonEmptyString
    task_record_id: str | None
    data_sample_id: str | None
    task_id: str | None
    stage: Stage
    lifecycle_state: LifecycleState
    date: TimestampString | None
    date_kind: NonEmptyString
    provider: str | None
    model: str | None
    encoder_system_prompt: str | None
    encoder_user_prompt: str | None
    encoder_output: NonEmptyString
    prompt_fidelity: PromptFidelity
    content_sha256: Sha256
    extraction_warning: str | None

    @field_validator("encoder_output")
    @classmethod
    def _require_nonblank_encoder_output(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("encoder_output must be nonblank")
        return value

    @field_validator("stage")
    @classmethod
    def _require_encoder_stage(cls, value: Stage) -> Stage:
        if value is not Stage.ENCODER:
            raise ValueError("encoder artifacts must have encoder stage")
        return value


class RequestRecord(FrozenModel):
    generation_id: NonEmptyString
    source_record_id: NonEmptyString
    dataset: DatasetName
    task_record_id: NonEmptyString
    task_id: NonEmptyString
    generation_mode: GenerationMode
    prompt_fidelity: PromptFidelity
    encoder_prompt_fidelity: PromptFidelity | None
    decoder_prompt_fidelity: PromptFidelity
    budget_mode: BudgetMode
    max_characters: Annotated[int, Field(ge=0)] | None
    source_project: NonEmptyString
    source_pool: NonEmptyString
    source_table: NonEmptyString
    source_file: NonEmptyString
    source_line_number: Annotated[int, Field(ge=1)]
    source_sample_id: NonEmptyString
    sample_idx: int | None
    run_id: str | None
    source_kind: str | None
    source_attempt_count: Annotated[int, Field(ge=0)]
    finish_reason: str | None
    response_finish_reason: str | None
    encoder_source_record_id: str | None
    encoder_config_id: str | None
    decoder_config_id: str | None
    encoder_prompt_template_id: str | None
    decoder_prompt_template_id: str | None
    encoder_provider: str | None
    encoder_model: str | None
    decoder_provider: str | None
    decoder_model: str | None
    encoder_reasoning_json: str | None
    decoder_reasoning_json: str | None
    encoder_temperature: float | None
    decoder_temperature: float | None
    encoder_top_p: float | None
    decoder_top_p: float | None
    encoder_max_tokens: int | None
    decoder_max_tokens: int | None
    key_values_json: NonEmptyString
    request_json: NonEmptyString
    response_json: NonEmptyString
    metadata_json: NonEmptyString
    hints_json: NonEmptyString
    extraction_warning: str | None


class TaskRecord(FrozenModel):
    task_record_id: NonEmptyString
    dataset: DatasetName
    source_variant: NonEmptyString
    task_id: NonEmptyString
    language: NonEmptyString
    dataset_id: str | None
    split: str | None
    data_sample_id: str | None
    source_digest: str | None
    dataset_revision: str | None
    evaluator_kind: NonEmptyString
    material_fidelity: TaskMaterialFidelity
    task_json: NonEmptyString
    content_sha256: Sha256


class ArtifactSummary(FrozenModel):
    path: NonEmptyString
    sha256: Sha256
    rows: Annotated[int, Field(ge=0)]
    schema_sha256: Sha256


class BuildManifest(FrozenModel):
    format: Literal["generation-corpus-v1"] = "generation-corpus-v1"
    adapter_name: NonEmptyString
    adapter_version: Annotated[int, Field(ge=1)]
    created_at: NonEmptyString
    source_manifest: NonEmptyString
    source_manifest_sha256: Sha256
    source_dump_created_at: NonEmptyString
    source_dump_pool_count: Annotated[int, Field(ge=0)]
    generations: ArtifactSummary
    source_records: ArtifactSummary
    encoder_artifacts: ArtifactSummary
    requests: ArtifactSummary
    tasks: ArtifactSummary


__all__ = [
    "ArtifactSummary",
    "BudgetMode",
    "BuildManifest",
    "DatasetName",
    "DecoderDescriptionSource",
    "DumpedPoolRow",
    "DumpHints",
    "DumpOutputKind",
    "EncoderArtifactRecord",
    "GenerationMode",
    "GenerationRecord",
    "LifecycleState",
    "PoolManifestEntry",
    "PoolSchema",
    "PoolSchemaColumn",
    "PromptFidelity",
    "RequestRecord",
    "SourceManifest",
    "SourceRecord",
    "Stage",
    "TaskMaterialFidelity",
    "TaskRecord",
]
