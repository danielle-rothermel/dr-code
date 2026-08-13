from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from pydantic import JsonValue

from drc_generation_corpus.models import (
    BudgetMode,
    DumpOutputKind,
    DumpedPoolRow,
    EncoderArtifactRecord,
    GenerationMode,
    GenerationRecord,
    LifecycleState,
    PromptFidelity,
    RequestRecord,
    SourceManifest,
    SourceRecord,
    Stage,
)
from drc_generation_corpus.pool_dump import (
    canonical_json,
    content_sha256,
    generation_id,
    iter_pool_rows,
    source_record_id,
)
from drc_generation_corpus.tasks.base import TaskAdapter
from drc_generation_corpus.tasks.code_eval_pro import (
    BIGCODEBENCH_LITE_PRO_DEFINITION,
    CLASS_EVAL_DEFINITION,
    HUMANEVAL_PRO_DEFINITION,
    MBPP_PRO_DEFINITION,
    BigCodeBenchLiteProTaskAdapter,
    ClassEvalTaskAdapter,
    CodeCompDatasetDefinition,
    HumanEvalProTaskAdapter,
    MbppProTaskAdapter,
)
from drc_generation_corpus.writer import CorpusWriter

_PROJECT: Final = "code_comp_v0"
_POOL_STAGES: Final[dict[str, Stage]] = {
    "direct_enc_t0": Stage.ENCODER,
    "encoder_pool_t1": Stage.ENCODER,
    "official_decoder_t0": Stage.DECODER,
    "reexport_shell_seed_decoder": Stage.DECODER,
    "reexport_shell_seed_encoder": Stage.ENCODER,
    "reexport_smoke_20260510_155435": Stage.ENCODER,
    "tstfix_20260510_034336": Stage.ENCODER,
}
_ALLOWED_POOL_COORDINATES: Final = {
    (_PROJECT, pool_name) for pool_name in _POOL_STAGES
}


def _optional_string(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value


def _prompt_fields(
    request: Mapping[str, JsonValue],
) -> tuple[PromptFidelity, str | None, str | None]:
    prompt = request.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(
                "persisted request prompt must be a non-empty array"
            )
        messages: dict[str, str] = {}
        for index, message_value in enumerate(prompt):
            if not isinstance(message_value, dict):
                raise ValueError(
                    f"request prompt message {index} must be an object"
                )
            role = message_value.get("role")
            content = message_value.get("content")
            if role not in {"system", "user"} or not isinstance(content, str):
                raise ValueError(
                    f"request prompt message {index} has unsupported role/content"
                )
            if role in messages:
                raise ValueError(
                    f"request prompt contains duplicate {role!r} message"
                )
            messages[role] = content
        if "user" not in messages:
            raise ValueError("persisted request prompt lacks a user message")
        return (
            PromptFidelity.EXACT_REQUEST,
            messages.get("system"),
            messages["user"],
        )

    if request == {
        "reason": "historical_migration",
        "unavailable": True,
    }:
        return PromptFidelity.UNAVAILABLE, None, None
    raise ValueError(
        "request is neither an exact prompt nor migration sentinel"
    )


def _output_text(row: DumpedPoolRow) -> tuple[str | None, bool]:
    if row.response_json is None:
        return None, False
    value = row.response_json.get("text")
    if value is None:
        return None, False
    if not isinstance(value, str):
        raise ValueError("response_json.text must be a string or null")
    return value, bool(value.strip())


def _lifecycle(
    *, stage: Stage, has_nonblank_output: bool, status: str | None
) -> LifecycleState:
    if has_nonblank_output:
        if stage is Stage.ENCODER:
            return LifecycleState.ENCODER_ONLY
        return LifecycleState.PENDING_VALIDATION
    if status == "failed":
        return LifecycleState.FAILED
    return LifecycleState.SEEDED


def _response_field(row: DumpedPoolRow, field: str) -> str | None:
    if row.response_json is None:
        return None
    return _optional_string(
        row.response_json.get(field), context=f"response_json.{field}"
    )


def _optional_number(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number or null")
    return float(value)


def _optional_integer(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer or null")
    return value


def _request_config(
    request: Mapping[str, JsonValue], *, fidelity: PromptFidelity
) -> tuple[
    str | None,
    str | None,
    str | None,
    float | None,
    float | None,
    int | None,
]:
    if fidelity is not PromptFidelity.EXACT_REQUEST:
        return None, None, None, None, None, None
    config_value = request.get("llm_config")
    if not isinstance(config_value, dict):
        raise ValueError("exact request lacks llm_config object")
    provider = _optional_string(
        config_value.get("provider"), context="request llm_config.provider"
    )
    model = _optional_string(
        config_value.get("model"), context="request llm_config.model"
    )
    if provider is None or model is None:
        raise ValueError("exact request llm_config lacks provider/model")
    reasoning_value = config_value.get("reasoning")
    if reasoning_value is not None and not isinstance(reasoning_value, dict):
        raise ValueError(
            "request llm_config.reasoning must be an object or null"
        )
    reasoning_json = (
        canonical_json(cast(JsonValue, reasoning_value))
        if reasoning_value is not None
        else None
    )
    return (
        provider,
        model,
        reasoning_json,
        _optional_number(
            config_value.get("temperature"),
            context="request llm_config.temperature",
        ),
        _optional_number(
            config_value.get("top_p"), context="request llm_config.top_p"
        ),
        _optional_integer(
            config_value.get("max_tokens"),
            context="request llm_config.max_tokens",
        ),
    )


def _warning(
    *,
    response_finish_reason: str | None,
    opaque_task_material: bool,
) -> str | None:
    warnings: list[str] = []
    if opaque_task_material:
        warnings.append("task_material_unavailable_for_opaque_source_variant")
    if response_finish_reason is None:
        warnings.append("response_finish_reason_unavailable")
    elif response_finish_reason.casefold() != "stop":
        warnings.append(
            f"non_stop_response_finish_reason:{response_finish_reason}"
        )
    return ";".join(warnings) or None


class _CodeCompAdapter:
    adapter_version = 1

    def __init__(
        self,
        *,
        definition: CodeCompDatasetDefinition,
        task_adapter: TaskAdapter,
        adapter_name: str,
    ) -> None:
        if task_adapter.dataset is not definition.dataset:
            raise ValueError(
                "task adapter dataset does not match code_comp definition"
            )
        self._definition = definition
        self._tasks = task_adapter
        self.adapter_name = adapter_name

    def populate(
        self,
        *,
        dump_directory: Path,
        source_manifest: SourceManifest,
        writer: CorpusWriter,
    ) -> None:
        actual_allowed_coordinates = {
            (entry.project_name, entry.pool_name)
            for entry in source_manifest.pools
            if (entry.project_name, entry.pool_name)
            in _ALLOWED_POOL_COORDINATES
        }
        if actual_allowed_coordinates != _ALLOWED_POOL_COORDINATES:
            missing = sorted(
                _ALLOWED_POOL_COORDINATES - actual_allowed_coordinates
            )
            raise ValueError(
                f"source manifest lacks required code_comp pools: {missing}"
            )

        for task_record in self._tasks.records():
            writer.add_task(task_record)

        seen_source_record_ids: set[str] = set()
        seen_source_sample_ids: set[str] = set()
        for entry in source_manifest.pools:
            coordinate = (entry.project_name, entry.pool_name)
            for source_line_number, row in enumerate(
                iter_pool_rows(dump_directory / entry.file_name, entry),
                start=1,
            ):
                parsed = self._definition.parse_data_sample_id(
                    row.key_values.get("data_sample_id")
                )
                if parsed is None:
                    continue
                if coordinate not in _ALLOWED_POOL_COORDINATES:
                    raise ValueError(
                        f"{self._definition.dataset.value} row appears in "
                        f"unapproved pool {coordinate!r}"
                    )
                record_id = source_record_id(row)
                if record_id in seen_source_record_ids:
                    raise ValueError(f"duplicate source record ID {record_id}")
                if row.sample_id in seen_source_sample_ids:
                    raise ValueError(
                        f"duplicate selected source sample ID {row.sample_id}"
                    )
                seen_source_record_ids.add(record_id)
                seen_source_sample_ids.add(row.sample_id)
                self._add_row(
                    row=row,
                    source_file=entry.file_name,
                    source_line_number=source_line_number,
                    task_id=parsed[0],
                    writer=writer,
                )

    def _add_row(
        self,
        *,
        row: DumpedPoolRow,
        source_file: str,
        source_line_number: int,
        task_id: str,
        writer: CorpusWriter,
    ) -> None:
        data_sample_id_value = row.key_values.get("data_sample_id")
        if not isinstance(data_sample_id_value, str):
            raise AssertionError("parsed data_sample_id is not a string")
        data_sample_id = data_sample_id_value
        stage = _POOL_STAGES[row.pool_name]
        source_variant = (
            self._definition.encoder_pool_source_variant
            if row.pool_name == "encoder_pool_t1"
            else self._definition.primary_source_variant
        )
        task_record = self._tasks.resolve(data_sample_id)
        opaque_task_material = (
            row.pool_name == "encoder_pool_t1"
            and not self._definition.encoder_pool_has_task_material
        )
        if opaque_task_material:
            if task_record is not None:
                raise ValueError(
                    "opaque encoder_pool_t1 source unexpectedly resolved by full ID"
                )
        elif task_record is None:
            raise ValueError(
                f"no pinned task material for full identity {data_sample_id}"
            )
        elif (
            task_record.task_id != task_id
            or task_record.source_variant != source_variant
        ):
            raise ValueError(
                f"pinned task identity disagrees with {data_sample_id}"
            )

        if stage is Stage.DECODER:
            self._validate_direct_decoder(row, data_sample_id=data_sample_id)

        text, has_nonblank_output = _output_text(row)
        if stage is Stage.DECODER and has_nonblank_output:
            if (
                row.hints.output_kind is not DumpOutputKind.CODE_TEXT
                or row.hints.output_json_path != "response_json.text"
            ):
                raise ValueError(
                    "nonblank direct decoder lacks code_text response path sentinel"
                )
        status = _optional_string(
            row.key_values.get("status"), context="key_values.status"
        )
        lifecycle_state = _lifecycle(
            stage=stage,
            has_nonblank_output=has_nonblank_output,
            status=status,
        )
        response_finish_reason = _response_field(row, "finish_reason")
        provider = _response_field(row, "provider")
        model = _response_field(row, "model")
        persisted_fidelity, system_prompt, user_prompt = _prompt_fields(
            row.request_json
        )
        prompt_fidelity = persisted_fidelity
        if (
            persisted_fidelity is PromptFidelity.UNAVAILABLE
            and stage is Stage.DECODER
            and task_record is not None
        ):
            prompt_fidelity = PromptFidelity.RECOVERED_TASK

        record_id = source_record_id(row)
        raw_response: JsonValue = cast(JsonValue, row.response_json)
        writer.add_source_record(
            SourceRecord(
                source_record_id=record_id,
                dataset=self._definition.dataset,
                source_variant=source_variant,
                data_sample_id=data_sample_id,
                task_id=task_id,
                source_project=row.project_name,
                source_pool=row.pool_name,
                source_table=row.table_name,
                source_file=source_file,
                source_line_number=source_line_number,
                source_sample_id=row.sample_id,
                sample_idx=row.sample_idx,
                run_id=row.run_id,
                stage=stage,
                lifecycle_state=lifecycle_state,
                date=row.created_at,
                date_kind=(
                    "pool_record_created_at"
                    if row.created_at is not None
                    else "unavailable"
                ),
                status=status,
                attempt_count=row.attempt_count,
                finish_reason=row.finish_reason,
                response_finish_reason=response_finish_reason,
                has_nonblank_output=has_nonblank_output,
                output_text=text,
                key_values_json=canonical_json(
                    cast(JsonValue, row.key_values)
                ),
                request_json=canonical_json(cast(JsonValue, row.request_json)),
                response_json=canonical_json(raw_response),
                metadata_json=canonical_json(
                    cast(JsonValue, row.metadata_json)
                ),
                hints_json=canonical_json(
                    cast(JsonValue, row.hints.model_dump(mode="json"))
                ),
            )
        )

        if not has_nonblank_output:
            return
        assert text is not None
        warning = _warning(
            response_finish_reason=response_finish_reason,
            opaque_task_material=opaque_task_material,
        )
        if stage is Stage.ENCODER:
            writer.add_encoder_artifact(
                EncoderArtifactRecord(
                    encoder_artifact_id=record_id,
                    source_record_id=record_id,
                    dataset=self._definition.dataset,
                    source_variant=source_variant,
                    task_record_id=(
                        task_record.task_record_id
                        if task_record is not None
                        else None
                    ),
                    data_sample_id=data_sample_id,
                    task_id=task_id,
                    stage=Stage.ENCODER,
                    lifecycle_state=LifecycleState.ENCODER_ONLY,
                    date=row.created_at,
                    date_kind=(
                        "pool_record_created_at"
                        if row.created_at is not None
                        else "unavailable"
                    ),
                    provider=provider,
                    model=model,
                    encoder_system_prompt=system_prompt,
                    encoder_user_prompt=user_prompt,
                    encoder_output=text,
                    prompt_fidelity=prompt_fidelity,
                    content_sha256=content_sha256(
                        system_prompt, user_prompt, text
                    ),
                    extraction_warning=warning,
                )
            )
            return

        if task_record is None:
            raise AssertionError(
                "direct decoder task material was not resolved"
            )
        generation_record_id = generation_id(row)
        is_partial = (
            response_finish_reason is None
            or response_finish_reason.casefold() != "stop"
        )
        writer.add_generation(
            GenerationRecord(
                generation_id=generation_record_id,
                source_record_id=record_id,
                dataset=self._definition.dataset,
                source_variant=source_variant,
                task_record_id=task_record.task_record_id,
                data_sample_id=data_sample_id,
                task_id=task_id,
                generation_mode=GenerationMode.DIRECT,
                stage=Stage.DECODER,
                lifecycle_state=LifecycleState.PENDING_VALIDATION,
                date=row.created_at,
                date_kind=(
                    "pool_record_created_at"
                    if row.created_at is not None
                    else "unavailable"
                ),
                provider=provider,
                model=model,
                encoder_provider=None,
                encoder_model=None,
                decoder_provider=provider,
                decoder_model=model,
                encoder_system_prompt=None,
                encoder_user_prompt=None,
                encoder_output=None,
                decoder_system_prompt=system_prompt,
                decoder_user_prompt=user_prompt,
                decoder_output=text,
                is_partial=is_partial,
                prompt_fidelity=prompt_fidelity,
                content_sha256=content_sha256(
                    None,
                    None,
                    None,
                    system_prompt,
                    user_prompt,
                    text,
                ),
                extraction_warning=warning,
            )
        )
        (
            request_provider,
            request_model,
            request_reasoning_json,
            request_temperature,
            request_top_p,
            request_max_tokens,
        ) = _request_config(row.request_json, fidelity=persisted_fidelity)
        writer.add_request(
            RequestRecord(
                generation_id=generation_record_id,
                source_record_id=record_id,
                dataset=self._definition.dataset,
                task_record_id=task_record.task_record_id,
                task_id=task_id,
                generation_mode=GenerationMode.DIRECT,
                prompt_fidelity=prompt_fidelity,
                encoder_prompt_fidelity=None,
                decoder_prompt_fidelity=prompt_fidelity,
                budget_mode=BudgetMode.NO_BUDGET,
                max_characters=None,
                source_project=row.project_name,
                source_pool=row.pool_name,
                source_table=row.table_name,
                source_file=source_file,
                source_line_number=source_line_number,
                source_sample_id=row.sample_id,
                sample_idx=row.sample_idx,
                run_id=row.run_id,
                source_kind=_optional_string(
                    row.metadata_json.get("source_kind"),
                    context="metadata_json.source_kind",
                ),
                source_attempt_count=row.attempt_count,
                finish_reason=row.finish_reason,
                response_finish_reason=response_finish_reason,
                encoder_source_record_id=None,
                encoder_config_id=None,
                decoder_config_id=_optional_string(
                    row.key_values.get("dec_llm_config_id"),
                    context="key_values.dec_llm_config_id",
                ),
                encoder_prompt_template_id=None,
                decoder_prompt_template_id=_optional_string(
                    row.key_values.get("dec_prompt_template_id"),
                    context="key_values.dec_prompt_template_id",
                ),
                encoder_provider=None,
                encoder_model=None,
                decoder_provider=request_provider,
                decoder_model=request_model,
                encoder_reasoning_json=None,
                decoder_reasoning_json=request_reasoning_json,
                encoder_temperature=None,
                decoder_temperature=request_temperature,
                encoder_top_p=None,
                decoder_top_p=request_top_p,
                encoder_max_tokens=None,
                decoder_max_tokens=request_max_tokens,
                key_values_json=canonical_json(
                    cast(JsonValue, row.key_values)
                ),
                request_json=canonical_json(cast(JsonValue, row.request_json)),
                response_json=canonical_json(raw_response),
                metadata_json=canonical_json(
                    cast(JsonValue, row.metadata_json)
                ),
                hints_json=canonical_json(
                    cast(JsonValue, row.hints.model_dump(mode="json"))
                ),
                extraction_warning=warning,
            )
        )

    @staticmethod
    def _validate_direct_decoder(
        row: DumpedPoolRow, *, data_sample_id: str
    ) -> None:
        expected = {
            "source_kind": "task_prompt",
            "enc_prompt_template_id": "sentinel/task_prompt",
            "enc_llm_config_id": "sentinel/non_llm",
            "enc_sample_id": f"sentinel/task_prompt/{data_sample_id}",
            "dec_prompt_template_id": "sentinel/official_prompt",
        }
        actual = {
            "source_kind": row.metadata_json.get("source_kind"),
            "enc_prompt_template_id": row.key_values.get(
                "enc_prompt_template_id"
            ),
            "enc_llm_config_id": row.key_values.get("enc_llm_config_id"),
            "enc_sample_id": row.key_values.get("enc_sample_id"),
            "dec_prompt_template_id": row.key_values.get(
                "dec_prompt_template_id"
            ),
        }
        if actual != expected:
            raise ValueError(
                "direct decoder sentinel mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )


class MbppProCodeCompAdapter(_CodeCompAdapter):
    def __init__(self, cache_directory: Path) -> None:
        super().__init__(
            definition=MBPP_PRO_DEFINITION,
            task_adapter=MbppProTaskAdapter(cache_directory),
            adapter_name="code_comp_mbpp_pro",
        )


class HumanEvalProCodeCompAdapter(_CodeCompAdapter):
    def __init__(self, cache_directory: Path) -> None:
        super().__init__(
            definition=HUMANEVAL_PRO_DEFINITION,
            task_adapter=HumanEvalProTaskAdapter(cache_directory),
            adapter_name="code_comp_humaneval_pro",
        )


class ClassEvalCodeCompAdapter(_CodeCompAdapter):
    def __init__(self, cache_directory: Path) -> None:
        super().__init__(
            definition=CLASS_EVAL_DEFINITION,
            task_adapter=ClassEvalTaskAdapter(cache_directory),
            adapter_name="code_comp_class_eval",
        )


class BigCodeBenchLiteProCodeCompAdapter(_CodeCompAdapter):
    def __init__(self, cache_directory: Path) -> None:
        super().__init__(
            definition=BIGCODEBENCH_LITE_PRO_DEFINITION,
            task_adapter=BigCodeBenchLiteProTaskAdapter(cache_directory),
            adapter_name="code_comp_bigcodebench_lite_pro",
        )


__all__ = [
    "BigCodeBenchLiteProCodeCompAdapter",
    "ClassEvalCodeCompAdapter",
    "HumanEvalProCodeCompAdapter",
    "MbppProCodeCompAdapter",
]
