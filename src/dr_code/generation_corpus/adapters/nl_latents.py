from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import Final, Literal

from pydantic import JsonValue, field_validator, model_validator

from dr_code.core.models import FrozenModel
from dr_code.generation_corpus.models import (
    BudgetMode,
    DatasetName,
    DumpedPoolRow,
    EncoderArtifactRecord,
    GenerationMode,
    GenerationRecord,
    LifecycleState,
    PoolManifestEntry,
    PromptFidelity,
    RequestRecord,
    SourceManifest,
    SourceRecord,
    Stage,
)
from dr_code.generation_corpus.pool_dump import (
    canonical_json,
    content_sha256,
    generation_id,
    iter_pool_rows,
    source_record_id,
)
from dr_code.generation_corpus.tasks.nl_latents import (
    NlLatentsFamily,
    NlLatentsLanguage,
    NlLatentsSplit,
    NlLatentsTaskAdapter,
    NlLatentsTaskCoordinate,
    NlLatentsTaskMapping,
)
from dr_code.generation_corpus.writer import CorpusWriter

_PROJECT: Final = "nl_latents"
_POOL: Final = "nl_latents"
_TABLE: Final = "pool_nl_latents_samples"
_FILE: Final = "nl_latents__nl_latents.jsonl.gz"
_SCHEMA: Final = "nl_latents"
_ROW_COUNT: Final = 192_333
_TASK_MAPPING_DIGEST: Final = (
    "29476987d8db1c646943b64c83219b08cb1033ebb8c0ec987daa697b54fbe695"
)
_KEY_COLUMNS: Final = (
    ("config_id", "text"),
    ("family", "text"),
    ("difficulty", "text"),
    ("split", "text"),
    ("language", "text"),
    ("budget", "text"),
    ("task_id", "text"),
    ("task_data_version", "text"),
    ("enc_model", "text"),
    ("dec_model", "text"),
    ("enc_reasoning_effort", "text"),
    ("dec_reasoning_effort", "text"),
    ("call_id", "text"),
    ("status", "text"),
)
_ACTIVE_REQUEST_MARKER: Final = {
    "reason": "historical_migration",
    "unavailable": True,
}
_FAILURE_CATEGORIES: Final = {
    "test_fail": "test_fail",
    "compile_or_parse_error": "compile_or_parse_error",
    "runtime_error": "runtime_error",
    "unknown_error": "unknown_error",
    "FailureCategory.TEST_FAIL": "test_fail",
    "FailureCategory.COMPILE_OR_PARSE_ERROR": "compile_or_parse_error",
    "FailureCategory.RUNTIME_ERROR": "runtime_error",
    "FailureCategory.UNKNOWN_ERROR": "unknown_error",
}
_CANDIDATE_WARNING: Final = (
    "candidate material only: evaluator, language runtime/toolchain, and "
    "resource policy are not pinned"
)


@verify(UNIQUE)
class NlLatentsStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"


@verify(UNIQUE)
class NlLatentsReasoningEffort(StrEnum):
    MINIMAL = "minimal"
    PROVIDER_DEFAULT = "provider_default"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NlLatentsKey(FrozenModel):
    config_id: str
    family: NlLatentsFamily
    difficulty: Literal["3", "4"]
    split: NlLatentsSplit
    language: NlLatentsLanguage
    budget: str
    task_id: str
    task_data_version: Literal[
        "tasks_v1_pre_resample_2026_02_10",
        "tasks_v2_resampled_2026_02_11",
    ]
    enc_model: str
    dec_model: str
    enc_reasoning_effort: NlLatentsReasoningEffort
    dec_reasoning_effort: NlLatentsReasoningEffort
    call_id: Literal["__legacy_null_key__:call_id"]
    status: NlLatentsStatus

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_smoke_language(cls, value: object) -> object:
        return "python" if value == "Python" else value

    @field_validator("config_id", "task_id", "enc_model", "dec_model")
    @classmethod
    def _require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("NL Latents key strings must be nonblank")
        return value

    @field_validator("budget")
    @classmethod
    def _validate_budget(cls, value: str) -> str:
        if not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise ValueError(f"invalid NL Latents character budget {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_stage_config(self) -> NlLatentsKey:
        if self.enc_model != self.dec_model:
            raise ValueError("encoder and decoder models differ")
        if self.enc_reasoning_effort is not self.dec_reasoning_effort:
            raise ValueError("encoder and decoder reasoning efforts differ")
        return self

    def task_coordinate(self) -> NlLatentsTaskCoordinate:
        difficulty: Literal[3, 4] = 3 if self.difficulty == "3" else 4
        return NlLatentsTaskCoordinate(
            task_data_version=self.task_data_version,
            family=self.family,
            difficulty=difficulty,
            split=self.split,
            language=self.language,
            task_id=self.task_id,
        )


@dataclass(frozen=True, slots=True)
class NlLatentsAdaptedRow:
    source_record: SourceRecord
    generation: GenerationRecord | None
    encoder_artifact: EncoderArtifactRecord | None
    request: RequestRecord | None
    task_mapping: NlLatentsTaskMapping
    normalized_failure_category: str | None
    old_eval_ready: bool


def normalize_failure_category(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("NL Latents failure category is not a string")
    try:
        return _FAILURE_CATEGORIES[raw]
    except KeyError as exc:
        raise ValueError(
            f"unknown NL Latents failure category {raw!r}"
        ) from exc


def _string(payload: dict[str, JsonValue], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"NL Latents {field_name} is not a string or null")
    return value


def _nonblank(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _integer(payload: dict[str, JsonValue], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"NL Latents {field_name} is not an integer or null")
    return value


def _boolean(payload: dict[str, JsonValue], field_name: str) -> bool | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"NL Latents {field_name} is not a boolean or null")
    return value


def _provider_model(value: str) -> tuple[str, str]:
    provider, separator, model = value.partition(":")
    if not separator or not provider or not model:
        raise ValueError(f"invalid NL Latents provider-model string {value!r}")
    return provider, model


def _payload_for_row(
    row: DumpedPoolRow, key: NlLatentsKey
) -> tuple[dict[str, JsonValue], str]:
    if key.status is NlLatentsStatus.ACTIVE:
        if row.response_json is None:
            raise ValueError("active NL Latents row has no response payload")
        if row.request_json != _ACTIVE_REQUEST_MARKER:
            raise ValueError(
                "active NL Latents row lacks the exact unavailable-request marker"
            )
        return row.response_json, "response_json"
    if row.response_json is not None:
        raise ValueError("pending NL Latents row has a response payload")
    if row.run_id is not None:
        raise ValueError("pending NL Latents row unexpectedly has a run_id")
    return row.request_json, "request_json"


def _validate_generation_payload(
    payload: dict[str, JsonValue], key: NlLatentsKey
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    bool,
]:
    encoder_prompt = _string(payload, "enc_prompt")
    description = _string(payload, "description")
    decoder_system = _string(payload, "dec_system")
    decoder_task = _string(payload, "dec_task")
    decoded_code = _string(payload, "decoded_code")
    validation_json = _string(payload, "validation_json")
    detail = _string(payload, "detail")

    actual_chars = _integer(payload, "actual_chars")
    budget_ok = _boolean(payload, "budget_ok")
    if _nonblank(description):
        if actual_chars is None:
            raise ValueError(
                "nonblank NL Latents description lacks actual_chars"
            )
        if actual_chars != len(description or ""):
            raise ValueError(
                "NL Latents actual_chars does not match description length"
            )
    if budget_ok is not None:
        if actual_chars is None:
            raise ValueError(
                "NL Latents budget_ok exists without actual_chars"
            )
        expected_budget_ok = actual_chars <= int(key.budget)
        if budget_ok is not expected_budget_ok:
            raise ValueError(
                "NL Latents budget_ok does not match actual_chars <= budget"
            )

    if _nonblank(decoded_code):
        required_prompts = {
            "enc_prompt": encoder_prompt,
            "description": description,
            "dec_system": decoder_system,
            "dec_task": decoder_task,
        }
        missing = [
            name
            for name, value in required_prompts.items()
            if not _nonblank(value)
        ]
        if missing:
            raise ValueError(
                "decoded NL Latents candidate lacks exact prompt evidence: "
                f"{missing!r}"
            )

    old_eval_ready = (
        key.status is NlLatentsStatus.ACTIVE
        and _nonblank(description)
        and _nonblank(decoded_code)
        and _nonblank(validation_json)
    )
    passed = _boolean(payload, "passed")
    failure = normalize_failure_category(payload.get("failure_category"))
    if key.status is NlLatentsStatus.ACTIVE:
        if passed is None:
            raise ValueError("active NL Latents row lacks boolean passed")
        if _nonblank(decoded_code) and not old_eval_ready:
            raise ValueError(
                "active decoded NL Latents candidate lacks historical validation"
            )
        if passed and failure is not None:
            raise ValueError("passing NL Latents row has a failure category")
    else:
        if (
            passed is not None
            or failure is not None
            or _nonblank(validation_json)
        ):
            raise ValueError(
                "pending NL Latents row contains evaluation evidence"
            )
        if not _nonblank(description) or not _nonblank(decoded_code):
            raise ValueError(
                "pending NL Latents row is not generation-complete"
            )
        if budget_ok is not True:
            raise ValueError("pending NL Latents row is not within budget")
        if detail != "pending docker validation":
            raise ValueError("pending NL Latents row has unexpected detail")

    if _nonblank(validation_json):
        try:
            validation = json.loads(validation_json or "")
        except json.JSONDecodeError as exc:
            raise ValueError(
                "NL Latents validation_json is invalid JSON"
            ) from exc
        if not isinstance(validation, dict):
            raise ValueError("NL Latents validation_json is not an object")

    return (
        encoder_prompt,
        description,
        decoder_system,
        decoder_task,
        decoded_code,
        validation_json,
        detail,
        old_eval_ready,
    )


def adapt_nl_latents_row(
    row: DumpedPoolRow,
    *,
    source_file: str,
    line_number: int,
    task_adapter: NlLatentsTaskAdapter,
) -> NlLatentsAdaptedRow:
    if (row.project_name, row.pool_name, row.table_name) != (
        _PROJECT,
        _POOL,
        _TABLE,
    ):
        raise ValueError("row is not from the NL Latents pool")
    if row.sample_idx is None or row.sample_idx < 0:
        raise ValueError("NL Latents sample_idx must be a nonnegative integer")
    if line_number < 1:
        raise ValueError("NL Latents source line number must be positive")
    key = NlLatentsKey.model_validate(row.key_values)
    payload, payload_branch = _payload_for_row(row, key)
    (
        encoder_prompt,
        description,
        decoder_system,
        decoder_task,
        decoded_code,
        validation_json,
        _detail,
        old_eval_ready,
    ) = _validate_generation_payload(payload, key)
    normalized_failure = normalize_failure_category(
        payload.get("failure_category")
    )

    coordinate = key.task_coordinate()
    task_mapping = task_adapter.resolve_coordinate(
        coordinate,
        validation_json=validation_json,
    )
    if task_mapping is None:
        raise ValueError(
            f"cannot resolve NL Latents task {coordinate.serialize()}"
        )
    if (
        key.family is not NlLatentsFamily.SMOKE
        and _nonblank(encoder_prompt)
        and not (encoder_prompt or "").endswith(task_mapping.code)
    ):
        raise ValueError(
            "NL Latents encoder prompt does not end with exact archived task code"
        )

    has_description = _nonblank(description)
    has_candidate = _nonblank(decoded_code)
    if has_candidate:
        lifecycle = (
            LifecycleState.EVALUATED
            if old_eval_ready
            else LifecycleState.PENDING_VALIDATION
        )
        stage = Stage.DECODER
        output_text = decoded_code
    elif has_description:
        lifecycle = LifecycleState.ENCODER_ONLY
        stage = Stage.ENCODER
        output_text = description
    else:
        lifecycle = LifecycleState.PRE_ENCODER_FAILURE
        stage = Stage.ENCODER
        output_text = decoded_code if decoded_code is not None else description

    row_source_record_id = source_record_id(row)
    source_record = SourceRecord(
        source_record_id=row_source_record_id,
        dataset=DatasetName.NL_LATENTS,
        source_variant=key.family.value,
        data_sample_id=None,
        task_id=key.task_id,
        source_project=row.project_name,
        source_pool=row.pool_name,
        source_table=row.table_name,
        source_file=source_file,
        source_line_number=line_number,
        source_sample_id=row.sample_id,
        sample_idx=row.sample_idx,
        run_id=row.run_id,
        stage=stage,
        lifecycle_state=lifecycle,
        date=row.created_at,
        date_kind="migration_import_created_at",
        status=key.status.value,
        attempt_count=row.attempt_count,
        finish_reason=row.finish_reason,
        response_finish_reason=_string(payload, "finish_reason"),
        has_nonblank_output=has_description or has_candidate,
        output_text=output_text,
        key_values_json=canonical_json(row.key_values),
        request_json=canonical_json(row.request_json),
        response_json=canonical_json(row.response_json),
        metadata_json=canonical_json(row.metadata_json),
        hints_json=canonical_json(row.hints.model_dump(mode="json")),
    )

    generation: GenerationRecord | None = None
    request: RequestRecord | None = None
    if has_candidate:
        encoder_provider, encoder_model = _provider_model(key.enc_model)
        decoder_provider, decoder_model = _provider_model(key.dec_model)
        if encoder_provider != decoder_provider:
            raise ValueError("encoder and decoder providers differ")
        row_generation_id = generation_id(row)
        generation = GenerationRecord(
            generation_id=row_generation_id,
            source_record_id=row_source_record_id,
            dataset=DatasetName.NL_LATENTS,
            source_variant=key.family.value,
            task_record_id=task_mapping.task_record_id,
            data_sample_id=None,
            task_id=key.task_id,
            generation_mode=GenerationMode.ENCODER_DECODER,
            stage=Stage.DECODER,
            lifecycle_state=lifecycle,
            date=row.created_at,
            date_kind="migration_import_created_at",
            provider=decoder_provider,
            model=decoder_model,
            encoder_provider=encoder_provider,
            encoder_model=encoder_model,
            decoder_provider=decoder_provider,
            decoder_model=decoder_model,
            encoder_system_prompt=None,
            encoder_user_prompt=encoder_prompt,
            encoder_output=description,
            decoder_system_prompt=decoder_system,
            decoder_user_prompt=decoder_task,
            decoder_output=decoded_code or "",
            is_partial=False,
            prompt_fidelity=PromptFidelity.EXACT_REQUEST,
            content_sha256=content_sha256(
                None,
                encoder_prompt,
                description,
                decoder_system,
                decoder_task,
                decoded_code,
            ),
            extraction_warning=_CANDIDATE_WARNING,
        )
        request = RequestRecord(
            generation_id=row_generation_id,
            source_record_id=row_source_record_id,
            dataset=DatasetName.NL_LATENTS,
            task_record_id=task_mapping.task_record_id,
            task_id=key.task_id,
            generation_mode=GenerationMode.ENCODER_DECODER,
            prompt_fidelity=PromptFidelity.EXACT_REQUEST,
            encoder_prompt_fidelity=PromptFidelity.EXACT_REQUEST,
            decoder_prompt_fidelity=PromptFidelity.EXACT_REQUEST,
            budget_mode=BudgetMode.BUDGET,
            max_characters=int(key.budget),
            source_project=row.project_name,
            source_pool=row.pool_name,
            source_table=row.table_name,
            source_file=source_file,
            source_line_number=line_number,
            source_sample_id=row.sample_id,
            sample_idx=row.sample_idx,
            run_id=row.run_id,
            source_kind=payload_branch,
            source_attempt_count=row.attempt_count,
            finish_reason=row.finish_reason,
            response_finish_reason=_string(payload, "finish_reason"),
            encoder_source_record_id=None,
            encoder_config_id=key.config_id,
            decoder_config_id=key.config_id,
            encoder_prompt_template_id=None,
            decoder_prompt_template_id=None,
            encoder_provider=encoder_provider,
            encoder_model=encoder_model,
            decoder_provider=decoder_provider,
            decoder_model=decoder_model,
            encoder_reasoning_json=canonical_json(
                {"effort": key.enc_reasoning_effort.value}
            ),
            decoder_reasoning_json=canonical_json(
                {"effort": key.dec_reasoning_effort.value}
            ),
            encoder_temperature=None,
            decoder_temperature=None,
            encoder_top_p=None,
            decoder_top_p=None,
            encoder_max_tokens=None,
            decoder_max_tokens=None,
            key_values_json=canonical_json(row.key_values),
            request_json=canonical_json(row.request_json),
            response_json=canonical_json(row.response_json),
            metadata_json=canonical_json(row.metadata_json),
            hints_json=canonical_json(row.hints.model_dump(mode="json")),
            extraction_warning=_CANDIDATE_WARNING,
        )

    encoder_artifact: EncoderArtifactRecord | None = None
    if has_description and not has_candidate:
        provider, model = _provider_model(key.enc_model)
        encoder_artifact = EncoderArtifactRecord(
            encoder_artifact_id=content_sha256(
                "nl_latents_encoder_artifact", row.sample_id
            ),
            source_record_id=row_source_record_id,
            dataset=DatasetName.NL_LATENTS,
            source_variant=key.family.value,
            task_record_id=task_mapping.task_record_id,
            data_sample_id=None,
            task_id=key.task_id,
            stage=Stage.ENCODER,
            lifecycle_state=LifecycleState.ENCODER_ONLY,
            date=row.created_at,
            date_kind="migration_import_created_at",
            provider=provider,
            model=model,
            encoder_system_prompt=None,
            encoder_user_prompt=encoder_prompt,
            encoder_output=description or "",
            prompt_fidelity=PromptFidelity.EXACT_REQUEST,
            content_sha256=content_sha256(
                None,
                encoder_prompt,
                description,
            ),
            extraction_warning=None,
        )

    return NlLatentsAdaptedRow(
        source_record=source_record,
        generation=generation,
        encoder_artifact=encoder_artifact,
        request=request,
        task_mapping=task_mapping,
        normalized_failure_category=normalized_failure,
        old_eval_ready=old_eval_ready,
    )


def _manifest_entry(source_manifest: SourceManifest) -> PoolManifestEntry:
    matches = [
        entry
        for entry in source_manifest.pools
        if (entry.project_name, entry.pool_name) == (_PROJECT, _POOL)
    ]
    if len(matches) != 1:
        raise ValueError(
            "source manifest must contain exactly one nl_latents/nl_latents pool"
        )
    entry = matches[0]
    actual_columns = tuple(
        (column.name, column.type)
        for column in entry.pool_schema_json.key_columns
    )
    actual = (
        entry.table_name,
        entry.file_name,
        entry.row_count,
        entry.dumped_row_count,
        entry.pool_schema_json.name,
        actual_columns,
        entry.original_status,
        entry.temporarily_started,
    )
    expected = (
        _TABLE,
        _FILE,
        _ROW_COUNT,
        _ROW_COUNT,
        _SCHEMA,
        _KEY_COLUMNS,
        "stopped",
        True,
    )
    if actual != expected:
        raise ValueError(
            "NL Latents manifest entry differs from the audited snapshot: "
            f"expected={expected!r}, actual={actual!r}"
        )
    return entry


@dataclass(frozen=True, slots=True)
class NlLatentsAdapter:
    archive_base: Path
    adapter_name: str = "nl_latents"
    adapter_version: int = 1

    def populate(
        self,
        *,
        dump_directory: Path,
        source_manifest: SourceManifest,
        writer: CorpusWriter,
    ) -> None:
        entry = _manifest_entry(source_manifest)
        task_adapter = NlLatentsTaskAdapter(self.archive_base)
        seen_sample_ids: set[str] = set()
        seen_key_indexes: set[tuple[str, int]] = set()
        counts: Counter[str] = Counter()

        for line_number, row in enumerate(
            iter_pool_rows(dump_directory / entry.file_name, entry), start=1
        ):
            if row.sample_id in seen_sample_ids:
                raise ValueError(
                    f"duplicate NL Latents sample_id {row.sample_id!r}"
                )
            seen_sample_ids.add(row.sample_id)
            if row.sample_idx is None:
                raise ValueError("NL Latents sample_idx is null")
            key_index = (canonical_json(row.key_values), row.sample_idx)
            if key_index in seen_key_indexes:
                raise ValueError(
                    "duplicate NL Latents (full key, sample_idx) identity"
                )
            seen_key_indexes.add(key_index)

            adapted = adapt_nl_latents_row(
                row,
                source_file=entry.file_name,
                line_number=line_number,
                task_adapter=task_adapter,
            )
            writer.add_source_record(adapted.source_record)
            counts["raw"] += 1
            counts[adapted.source_record.lifecycle_state.value] += 1
            if adapted.generation is not None:
                if adapted.request is None:
                    raise RuntimeError(
                        "canonical NL Latents generation lacks request "
                        "provenance record"
                    )
                writer.add_generation(adapted.generation)
                writer.add_request(adapted.request)
                counts["canonical"] += 1
                if adapted.old_eval_ready:
                    counts["old_eval_ready"] += 1
            if adapted.encoder_artifact is not None:
                writer.add_encoder_artifact(adapted.encoder_artifact)

        task_adapter.assert_ambiguous_resolutions_validated()
        task_records = tuple(task_adapter.records())
        expected_counts = {
            "raw": _ROW_COUNT,
            "canonical": 191_462,
            "old_eval_ready": 191_333,
            LifecycleState.EVALUATED.value: 191_333,
            LifecycleState.PENDING_VALIDATION.value: 129,
            LifecycleState.ENCODER_ONLY.value: 526,
            LifecycleState.PRE_ENCODER_FAILURE.value: 345,
        }
        actual_counts = {name: counts[name] for name in expected_counts}
        if actual_counts != expected_counts:
            raise ValueError(
                "NL Latents extraction counts differ from the audited snapshot: "
                f"expected={expected_counts!r}, actual={actual_counts!r}"
            )
        if len(task_records) != 294:
            raise ValueError(
                "NL Latents task mapping count differs from the audited "
                f"snapshot: expected=294, actual={len(task_records)}"
            )
        task_mapping_digest = content_sha256(
            sorted(record.task_record_id for record in task_records)
        )
        if task_mapping_digest != _TASK_MAPPING_DIGEST:
            raise ValueError(
                "NL Latents task mapping differs from the audited snapshot: "
                f"expected={_TASK_MAPPING_DIGEST}, "
                f"actual={task_mapping_digest}"
            )
        for task_record in task_records:
            writer.add_task(task_record)


__all__ = [
    "NlLatentsAdaptedRow",
    "NlLatentsAdapter",
    "NlLatentsKey",
    "NlLatentsReasoningEffort",
    "NlLatentsStatus",
    "adapt_nl_latents_row",
    "normalize_failure_category",
]
