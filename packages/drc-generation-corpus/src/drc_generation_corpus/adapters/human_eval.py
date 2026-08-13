from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from pathlib import Path
from typing import Final, cast

from pydantic import JsonValue

from drc_generation_corpus.models import (
    BudgetMode,
    DatasetName,
    DumpedPoolRow,
    DumpOutputKind,
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
    TaskRecord,
)
from drc_generation_corpus.pool_dump import (
    canonical_json,
    content_sha256,
    generation_id,
    iter_pool_rows,
    source_record_id,
)
from drc_generation_corpus.tasks.human_eval import (
    HumanEvalTaskAdapter,
    parse_human_eval_data_sample_id,
)
from drc_generation_corpus.writer import CorpusWriter

_DATE_KIND: Final = "created_at"
_ENCODER_REFERENCE_RE: Final = re.compile(
    r"^encoder_pool/(?P<pool>[^/]+)/(?P<sample_id>[^/]+)$"
)
_UNRESOLVED_ENCODER_SOURCE_RECORD_ID: Final = (
    "code_comp_v0:tde_20260510_0345:dca9c9034e234318b9e1b5a13703bf7b"
)
_BUDGET_ID_RE: Final = re.compile(r"(?:^|/)var_budget=(?P<value>\d+)(?:/|$)")
_BUDGET_PROMPT_RE: Final = re.compile(
    r"\busing at most (?P<value>\d+) characters?\b", re.IGNORECASE
)
_POOL_STAGES: Final[dict[tuple[str, str], Stage]] = {
    ("code_comp_t1", "budget_dec_v0_output_only"): Stage.DECODER,
    ("code_comp_t1", "budget_dec_v0"): Stage.DECODER,
    ("code_comp_t1", "budget_dec_v0_size6"): Stage.DECODER,
    ("code_comp_t1", "dec_v0_orig_docstring_output_only"): Stage.DECODER,
    ("code_comp_t1", "dec_v0_orig_docstring"): Stage.DECODER,
    ("code_comp_t1", "dec_v0_orig"): Stage.DECODER,
    ("code_comp_t1", "decoder_t1_smoke_3"): Stage.DECODER,
    ("code_comp_t1", "decoder_t2_smoke_3"): Stage.DECODER,
    ("code_comp_t1", "budget_enc_v0_output_only"): Stage.ENCODER,
    ("code_comp_t1", "budget_enc_v0"): Stage.ENCODER,
    ("code_comp_t1", "budget_enc_v0_size6"): Stage.ENCODER,
    ("code_comp_t1", "encoder_t1_smoke_3"): Stage.ENCODER,
    ("code_comp_v0", "direct_enc_t0"): Stage.ENCODER,
    ("code_comp_v0", "encoder_pool_t1"): Stage.ENCODER,
    ("code_comp_v0", "encoder_s6t3g2_v0"): Stage.ENCODER,
    ("code_comp_v0", "reexport_seed_encoder"): Stage.ENCODER,
    ("code_comp_v0", "reexport_shell_seed_encoder"): Stage.ENCODER,
    ("code_comp_v0", "reexport_smoke_20260510_155435"): Stage.ENCODER,
    ("code_comp_v0", "tstfix_20260510_034336"): Stage.ENCODER,
    ("code_comp_v0", "official_decoder_t0"): Stage.DECODER,
    ("code_comp_v0", "reexport_seed_decoder_from_encoder"): Stage.DECODER,
    ("code_comp_v0", "reexport_seed_decoder_simple"): Stage.DECODER,
    ("code_comp_v0", "reexport_shell_seed_decoder"): Stage.DECODER,
    ("code_comp_v0", "tde_20260510_0345"): Stage.DECODER,
    ("code_comp_v0", "tds_20260510_0344"): Stage.DECODER,
}


@verify(UNIQUE)
class _SourceKind(StrEnum):
    DOCSTRING_ONLY = "docstring_only"
    ENCODER_SAMPLE = "encoder_sample"
    ORIGINAL_HUMANEVAL_PROMPT = "original_humaneval_prompt"
    TASK_PROMPT = "task_prompt"


@dataclass(frozen=True, slots=True)
class _Id:
    data_sample_id: str
    task_id: str
    source_variant: str


@dataclass(frozen=True, slots=True)
class _EncoderReference:
    project_name: str
    pool_name: str
    sample_id: str


@dataclass(frozen=True, slots=True)
class _UnresolvedEncoderReference:
    pass


_UNRESOLVED_ENCODER_REFERENCE: Final = _UnresolvedEncoderReference()


@dataclass(frozen=True, slots=True)
class _LocatedRow:
    entry: PoolManifestEntry
    line_number: int
    row: DumpedPoolRow
    identity: _Id


@dataclass(frozen=True, slots=True)
class _BufferedDecoder:
    entry: PoolManifestEntry
    line_number: int
    row: DumpedPoolRow
    identity: _Id
    decoder_output: str


@dataclass(frozen=True, slots=True)
class _PromptProjection:
    system_prompt: str | None
    user_prompt: str | None
    fidelity: PromptFidelity
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _RequestControls:
    provider: str | None = None
    model: str | None = None
    reasoning_json: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class HumanEvalAdapter:
    """Extract the original historical HumanEval pool population."""

    adapter_name: str = "legacy_human_eval"
    adapter_version: int = 1

    def __init__(self, tasks: HumanEvalTaskAdapter) -> None:
        self._tasks = tasks

    def populate(
        self,
        *,
        dump_directory: Path,
        source_manifest: SourceManifest,
        writer: CorpusWriter,
    ) -> None:
        self._populate_rows(dump_directory, source_manifest, writer)
        for task in self._tasks.records():
            writer.add_task(task)

    def _populate_rows(
        self,
        dump_directory: Path,
        source_manifest: SourceManifest,
        writer: CorpusWriter,
    ) -> None:
        references: set[_EncoderReference] = set()
        source_ids: set[str] = set()
        generation_ids: set[str] = set()
        all_encoders: dict[_EncoderReference, _LocatedRow] = {}
        buffered_decoders: list[_BufferedDecoder] = []
        selected_rows = 0

        for entry in source_manifest.pools:
            for line_number, row in enumerate(
                iter_pool_rows(dump_directory / entry.file_name, entry),
                start=1,
            ):
                identity = _selected_identity(row)
                if identity is None:
                    continue
                selected_rows += 1
                row_source_id = source_record_id(row)
                if row_source_id in source_ids:
                    raise ValueError(
                        f"duplicate HumanEval source identity {row_source_id!r}"
                    )
                source_ids.add(row_source_id)

                stage = _stage(row)
                output = _response_text(row.response_json)
                if stage is Stage.DECODER and _is_nonblank(output):
                    row_generation_id = generation_id(row)
                    if row_generation_id in generation_ids:
                        raise ValueError(
                            "duplicate HumanEval generation identity "
                            f"{row_generation_id!r}"
                        )
                    generation_ids.add(row_generation_id)
                    reference = _encoder_reference(row)
                    if isinstance(reference, _EncoderReference):
                        references.add(reference)
                    assert output is not None
                    buffered_decoders.append(
                        _BufferedDecoder(
                            entry=entry,
                            line_number=line_number,
                            row=row,
                            identity=identity,
                            decoder_output=output,
                        )
                    )

                _require_task(self._tasks, identity)
                writer.add_source_record(
                    _source_record(
                        entry,
                        line_number=line_number,
                        row=row,
                        identity=identity,
                    )
                )
                if stage is not Stage.ENCODER:
                    continue
                reference = _EncoderReference(
                    row.project_name, row.pool_name, row.sample_id
                )
                if reference in all_encoders:
                    raise ValueError(
                        f"duplicate resolved encoder source {reference!r}"
                    )
                all_encoders[reference] = _LocatedRow(
                    entry=entry,
                    line_number=line_number,
                    row=row,
                    identity=identity,
                )

        if selected_rows == 0:
            raise ValueError("source dumps contain no HumanEval rows")

        encoders = {
            reference: all_encoders[reference] for reference in references
        }
        missing = sorted(
            references.difference(encoders),
            key=lambda value: (
                value.project_name,
                value.pool_name,
                value.sample_id,
            ),
        )
        if missing:
            raise ValueError(
                "HumanEval decoder encoder references do not resolve exactly: "
                f"{missing[:5]!r}"
            )

        for reference, located in all_encoders.items():
            if reference in references:
                continue
            output = _response_text(located.row.response_json)
            if not _is_nonblank(output):
                continue
            assert output is not None
            writer.add_encoder_artifact(
                _encoder_artifact(
                    located.row,
                    identity=located.identity,
                    task=_require_task(self._tasks, located.identity),
                    output=output,
                )
            )

        for buffered in buffered_decoders:
            task = _require_task(self._tasks, buffered.identity)
            generation, request = _generation_and_request(
                buffered.entry,
                line_number=buffered.line_number,
                row=buffered.row,
                identity=buffered.identity,
                task=task,
                decoder_output=buffered.decoder_output,
                encoders=encoders,
            )
            writer.add_generation(generation)
            writer.add_request(request)


def _selected_identity(row: DumpedPoolRow) -> _Id | None:
    candidates: list[tuple[str, object]] = [
        ("key_values.data_sample_id", row.key_values.get("data_sample_id")),
        (
            "metadata_json.data_sample_id",
            row.metadata_json.get("data_sample_id"),
        ),
    ]
    human_eval_values = [
        (location, value)
        for location, value in candidates
        if isinstance(value, str) and value.startswith("human_eval/")
    ]
    if not human_eval_values:
        return None
    selected = human_eval_values[0][1]
    if not isinstance(selected, str):
        raise AssertionError("selected data_sample_id is not a string")
    for location, value in candidates:
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"row {source_record_id(row)} has non-string {location}"
            )
        if value != selected:
            raise ValueError(
                f"row {source_record_id(row)} has conflicting "
                f"data_sample_id values: {selected!r} != {value!r}"
            )
    task_id, source_digest = parse_human_eval_data_sample_id(selected)
    if row.hints.human_eval_pro_task_id is not None:
        raise ValueError(
            f"row {source_record_id(row)} is both HumanEval and HumanEval Pro"
        )
    if row.hints.human_eval_task_id != task_id:
        raise ValueError(
            f"row {source_record_id(row)} HumanEval hint conflicts with "
            f"persisted identity: {row.hints.human_eval_task_id!r} != "
            f"{task_id!r}"
        )
    if selected == f"human_eval/{task_id}":
        source_variant = "unqualified_task"
    elif source_digest is None:
        source_variant = "gt_solution"
    else:
        source_variant = f"gt_solution@{source_digest}"
    return _Id(selected, task_id, source_variant)


def _stage(row: DumpedPoolRow) -> Stage:
    coordinate = (row.project_name, row.pool_name)
    try:
        stage = _POOL_STAGES[coordinate]
    except KeyError as error:
        raise ValueError(
            f"row {source_record_id(row)} has unknown HumanEval pool "
            f"coordinate {coordinate!r}"
        ) from error
    if stage is Stage.DECODER:
        if row.hints.output_kind is DumpOutputKind.CODE_TEXT:
            if row.hints.output_json_path == "response_json.text":
                return stage
            raise ValueError(
                f"row {source_record_id(row)} has unsupported code output "
                f"path {row.hints.output_json_path!r}"
            )
        if (
            row.hints.output_kind is DumpOutputKind.NOT_CODE
            and row.hints.output_json_path is None
        ):
            return stage
        raise ValueError(
            f"row {source_record_id(row)} has output hints inconsistent "
            "with its decoder pool"
        )
    if (
        row.hints.output_kind is not DumpOutputKind.NOT_CODE
        or row.hints.output_json_path is not None
    ):
        raise ValueError(
            f"row {source_record_id(row)} has output hints inconsistent "
            "with its encoder pool"
        )
    return stage


def _encoder_reference(
    row: DumpedPoolRow,
) -> _EncoderReference | _UnresolvedEncoderReference | None:
    source_kind = _optional_string(row.metadata_json.get("source_kind"))
    raw_metadata = row.metadata_json.get("source_sample_id")
    raw_key = row.key_values.get("source_sample_id")
    values = [value for value in (raw_metadata, raw_key) if value is not None]
    if any(not isinstance(value, str) for value in values):
        raise ValueError(
            f"row {source_record_id(row)} has a non-string source_sample_id"
        )
    references = {value for value in values if isinstance(value, str)}
    if len(references) > 1:
        raise ValueError(
            f"row {source_record_id(row)} has conflicting encoder references"
        )
    if source_kind in (
        _SourceKind.DOCSTRING_ONLY,
        _SourceKind.ORIGINAL_HUMANEVAL_PROMPT,
        _SourceKind.TASK_PROMPT,
    ):
        if references and not next(iter(references)).startswith(
            f"{source_kind}/"
        ):
            raise ValueError(
                f"row {source_record_id(row)} has a mismatched direct "
                "source_sample_id"
            )
        return None
    if source_kind != _SourceKind.ENCODER_SAMPLE:
        if references:
            raise ValueError(
                f"row {source_record_id(row)} has encoder lineage without "
                "source_kind='encoder_sample'"
            )
        return None
    if not references:
        if source_record_id(row) == _UNRESOLVED_ENCODER_SOURCE_RECORD_ID:
            return _UNRESOLVED_ENCODER_REFERENCE
        raise ValueError(
            f"row {source_record_id(row)} claims encoder lineage without "
            "source_sample_id"
        )
    raw_reference = next(iter(references))
    match = _ENCODER_REFERENCE_RE.fullmatch(raw_reference)
    if match is None:
        raise ValueError(
            f"row {source_record_id(row)} has malformed encoder reference "
            f"{raw_reference!r}"
        )
    pool_name = match.group("pool")
    metadata_pool = row.metadata_json.get("source_pool_name")
    if metadata_pool is not None and metadata_pool != pool_name:
        raise ValueError(
            f"row {source_record_id(row)} encoder pool conflicts with its "
            f"source_sample_id: {metadata_pool!r} != {pool_name!r}"
        )
    return _EncoderReference(
        project_name=row.project_name,
        pool_name=pool_name,
        sample_id=match.group("sample_id"),
    )


def _source_record(
    entry: PoolManifestEntry,
    *,
    line_number: int,
    row: DumpedPoolRow,
    identity: _Id,
) -> SourceRecord:
    stage = _stage(row)
    output = _response_text(row.response_json)
    status = _status(row)
    return SourceRecord(
        source_record_id=source_record_id(row),
        dataset=DatasetName.HUMAN_EVAL,
        source_variant=identity.source_variant,
        data_sample_id=identity.data_sample_id,
        task_id=identity.task_id,
        source_project=row.project_name,
        source_pool=row.pool_name,
        source_table=row.table_name,
        source_file=entry.file_name,
        source_line_number=line_number,
        source_sample_id=row.sample_id,
        sample_idx=row.sample_idx,
        run_id=row.run_id,
        stage=stage,
        lifecycle_state=_lifecycle(stage, output=output, status=status),
        date=row.created_at,
        date_kind=_DATE_KIND,
        status=status,
        attempt_count=row.attempt_count,
        finish_reason=row.finish_reason,
        response_finish_reason=_response_finish_reason(row.response_json),
        has_nonblank_output=_is_nonblank(output),
        output_text=output,
        key_values_json=canonical_json(row.key_values),
        request_json=canonical_json(row.request_json),
        response_json=canonical_json(row.response_json),
        metadata_json=canonical_json(row.metadata_json),
        hints_json=canonical_json(row.hints.model_dump(mode="json")),
    )


def _encoder_artifact(
    row: DumpedPoolRow,
    *,
    identity: _Id,
    task: TaskRecord,
    output: str,
) -> EncoderArtifactRecord:
    prompt = _prompt_projection(row.request_json, recovered_prompt=None)
    controls = _request_controls(row.request_json)
    provider, model = _response_model(row.response_json, controls)
    return EncoderArtifactRecord(
        encoder_artifact_id=f"encoder:{generation_id(row)}",
        source_record_id=source_record_id(row),
        dataset=DatasetName.HUMAN_EVAL,
        source_variant=identity.source_variant,
        task_record_id=task.task_record_id,
        data_sample_id=identity.data_sample_id,
        task_id=identity.task_id,
        stage=Stage.ENCODER,
        lifecycle_state=LifecycleState.ENCODER_ONLY,
        date=row.created_at,
        date_kind=_DATE_KIND,
        provider=provider,
        model=model,
        encoder_system_prompt=prompt.system_prompt,
        encoder_user_prompt=prompt.user_prompt,
        encoder_output=output,
        prompt_fidelity=prompt.fidelity,
        content_sha256=content_sha256(
            prompt.system_prompt, prompt.user_prompt, output
        ),
        extraction_warning=_warnings(prompt.warnings),
    )


def _generation_and_request(
    entry: PoolManifestEntry,
    *,
    line_number: int,
    row: DumpedPoolRow,
    identity: _Id,
    task: TaskRecord,
    decoder_output: str,
    encoders: Mapping[_EncoderReference, _LocatedRow],
) -> tuple[GenerationRecord, RequestRecord]:
    if row.created_at is None:
        raise ValueError(
            f"canonical HumanEval row {source_record_id(row)} has no timestamp"
        )
    task_prompt = _task_prompt(task)
    decoder_prompt = _prompt_projection(
        row.request_json, recovered_prompt=task_prompt
    )
    decoder_controls = _request_controls(row.request_json)
    decoder_provider, decoder_model = _response_model(
        row.response_json, decoder_controls
    )
    reference = _encoder_reference(row)
    warnings = set(decoder_prompt.warnings)
    encoder_row: DumpedPoolRow | None = None
    encoder_prompt: _PromptProjection | None = None
    encoder_controls = _RequestControls()
    encoder_provider: str | None = None
    encoder_model: str | None = None
    encoder_output: str | None = None
    encoder_config_id: str | None = None
    encoder_prompt_template_id: str | None = None
    max_characters: int | None = None
    if reference is None:
        generation_mode = GenerationMode.DIRECT
        budget_mode = BudgetMode.NO_BUDGET
    elif isinstance(reference, _UnresolvedEncoderReference):
        generation_mode = GenerationMode.UNRESOLVED_ENCODER
        budget_mode = BudgetMode.UNRESOLVED
        warnings.add("encoder_reference_unavailable")
    else:
        located_encoder = encoders.get(reference)
        if located_encoder is None:
            raise ValueError(
                f"row {source_record_id(row)} encoder reference is unresolved"
            )
        encoder_row = located_encoder.row
        _validate_encoder_lineage(
            decoder=row,
            decoder_identity=identity,
            encoder=located_encoder,
        )
        encoder_output = _response_text(encoder_row.response_json)
        if not _is_nonblank(encoder_output):
            raise ValueError(
                f"encoder {source_record_id(encoder_row)} has no nonblank output"
            )
        encoder_prompt = _prompt_projection(
            encoder_row.request_json, recovered_prompt=None
        )
        encoder_controls = _request_controls(encoder_row.request_json)
        encoder_provider, encoder_model = _response_model(
            encoder_row.response_json, encoder_controls
        )
        encoder_config_id = _optional_string(
            encoder_row.key_values.get("llm_config_id")
        )
        encoder_prompt_template_id = _optional_string(
            encoder_row.key_values.get("prompt_template_id")
        )
        max_characters = _budget_value(
            encoder_prompt_template_id, encoder_prompt.user_prompt
        )
        budget_mode = (
            BudgetMode.BUDGET
            if max_characters is not None
            else BudgetMode.NO_BUDGET
        )
        generation_mode = GenerationMode.ENCODER_DECODER
        warnings.update(encoder_prompt.warnings)

    fidelity = _aggregate_fidelity(encoder_prompt, decoder_prompt)
    row_generation_id = generation_id(row)
    decoder_config_id = _first_string(
        row.key_values.get("dec_llm_config_id"),
        row.key_values.get("llm_config_id"),
    )
    decoder_prompt_template_id = _first_string(
        row.key_values.get("dec_prompt_template_id"),
        row.key_values.get("prompt_template_id"),
    )
    warning = _warnings(tuple(sorted(warnings)))
    response_finish_reason = _response_finish_reason(row.response_json)
    generation = GenerationRecord(
        generation_id=row_generation_id,
        source_record_id=source_record_id(row),
        dataset=DatasetName.HUMAN_EVAL,
        source_variant=identity.source_variant,
        task_record_id=task.task_record_id,
        data_sample_id=identity.data_sample_id,
        task_id=identity.task_id,
        generation_mode=generation_mode,
        stage=Stage.DECODER,
        lifecycle_state=LifecycleState.PENDING_VALIDATION,
        date=row.created_at,
        date_kind=_DATE_KIND,
        provider=decoder_provider,
        model=decoder_model,
        encoder_provider=encoder_provider,
        encoder_model=encoder_model,
        decoder_provider=decoder_provider,
        decoder_model=decoder_model,
        encoder_system_prompt=(
            encoder_prompt.system_prompt if encoder_prompt else None
        ),
        encoder_user_prompt=(
            encoder_prompt.user_prompt if encoder_prompt else None
        ),
        encoder_output=encoder_output,
        decoder_system_prompt=decoder_prompt.system_prompt,
        decoder_user_prompt=decoder_prompt.user_prompt,
        decoder_output=decoder_output,
        is_partial=_is_partial(row.finish_reason, response_finish_reason),
        prompt_fidelity=fidelity,
        content_sha256=content_sha256(
            encoder_prompt.system_prompt if encoder_prompt else None,
            encoder_prompt.user_prompt if encoder_prompt else None,
            encoder_output,
            decoder_prompt.system_prompt,
            decoder_prompt.user_prompt,
            decoder_output,
        ),
        extraction_warning=warning,
    )
    request = RequestRecord(
        generation_id=row_generation_id,
        source_record_id=source_record_id(row),
        dataset=DatasetName.HUMAN_EVAL,
        task_record_id=task.task_record_id,
        task_id=identity.task_id,
        generation_mode=generation_mode,
        prompt_fidelity=fidelity,
        encoder_prompt_fidelity=(
            encoder_prompt.fidelity if encoder_prompt else None
        ),
        decoder_prompt_fidelity=decoder_prompt.fidelity,
        budget_mode=budget_mode,
        max_characters=max_characters,
        source_project=row.project_name,
        source_pool=row.pool_name,
        source_table=row.table_name,
        source_file=entry.file_name,
        source_line_number=line_number,
        source_sample_id=row.sample_id,
        sample_idx=row.sample_idx,
        run_id=row.run_id,
        source_kind=_optional_string(row.metadata_json.get("source_kind")),
        source_attempt_count=row.attempt_count,
        finish_reason=row.finish_reason,
        response_finish_reason=response_finish_reason,
        encoder_source_record_id=(
            source_record_id(encoder_row) if encoder_row else None
        ),
        encoder_config_id=encoder_config_id,
        decoder_config_id=decoder_config_id,
        encoder_prompt_template_id=encoder_prompt_template_id,
        decoder_prompt_template_id=decoder_prompt_template_id,
        encoder_provider=encoder_provider,
        encoder_model=encoder_model,
        decoder_provider=decoder_provider,
        decoder_model=decoder_model,
        encoder_reasoning_json=encoder_controls.reasoning_json,
        decoder_reasoning_json=decoder_controls.reasoning_json,
        encoder_temperature=encoder_controls.temperature,
        decoder_temperature=decoder_controls.temperature,
        encoder_top_p=encoder_controls.top_p,
        decoder_top_p=decoder_controls.top_p,
        encoder_max_tokens=encoder_controls.max_tokens,
        decoder_max_tokens=decoder_controls.max_tokens,
        key_values_json=canonical_json(row.key_values),
        request_json=canonical_json(row.request_json),
        response_json=canonical_json(row.response_json),
        metadata_json=canonical_json(row.metadata_json),
        hints_json=canonical_json(row.hints.model_dump(mode="json")),
        extraction_warning=warning,
    )
    return generation, request


def _validate_encoder_lineage(
    *,
    decoder: DumpedPoolRow,
    decoder_identity: _Id,
    encoder: _LocatedRow,
) -> None:
    if encoder.identity.data_sample_id != decoder_identity.data_sample_id:
        raise ValueError(
            f"row {source_record_id(decoder)} and encoder "
            f"{source_record_id(encoder.row)} have conflicting data_sample_id"
        )
    encoder_output = _response_text(encoder.row.response_json)
    embedded_output = _first_string(
        decoder.metadata_json.get("source_text"),
        _nested_string(
            decoder.metadata_json.get("source_sample_payload"), "text"
        ),
    )
    if embedded_output is not None and embedded_output != encoder_output:
        raise ValueError(
            f"row {source_record_id(decoder)} embeds encoder output that does "
            "not match its referenced encoder row"
        )
    _validate_lineage_field(
        decoder,
        metadata_field="enc_llm_config_id",
        encoder_value=_optional_string(
            encoder.row.key_values.get("llm_config_id")
        ),
        label="encoder config",
    )
    _validate_lineage_field(
        decoder,
        metadata_field="enc_prompt_template_id",
        encoder_value=_optional_string(
            encoder.row.key_values.get("prompt_template_id")
        ),
        label="encoder prompt",
    )


def _validate_lineage_field(
    decoder: DumpedPoolRow,
    *,
    metadata_field: str,
    encoder_value: str | None,
    label: str,
) -> None:
    expected = _optional_string(decoder.metadata_json.get(metadata_field))
    if expected is not None and expected != encoder_value:
        raise ValueError(
            f"row {source_record_id(decoder)} {label} mismatch: "
            f"{expected!r} != {encoder_value!r}"
        )


def _prompt_projection(
    request_json: Mapping[str, JsonValue], *, recovered_prompt: str | None
) -> _PromptProjection:
    unavailable = request_json.get("unavailable") is True
    prompt = request_json.get("prompt")
    if unavailable:
        if prompt is not None:
            raise ValueError("unavailable request also contains a prompt")
        if recovered_prompt is None:
            return _PromptProjection(
                None,
                None,
                PromptFidelity.UNAVAILABLE,
                ("original_request_unavailable",),
            )
        return _PromptProjection(
            None,
            recovered_prompt,
            PromptFidelity.RECOVERED_TASK,
            ("original_request_unavailable",),
        )
    if isinstance(prompt, str) and prompt:
        return _PromptProjection(None, prompt, PromptFidelity.EXACT_REQUEST)
    if not isinstance(prompt, list):
        return _PromptProjection(
            None,
            None,
            PromptFidelity.UNAVAILABLE,
            ("structured_prompt_unavailable",),
        )
    messages: dict[str, str] = {}
    for item in prompt:
        if not isinstance(item, Mapping):
            return _unavailable_structured_prompt()
        role = item.get("role")
        content = item.get("content")
        if (
            role not in {"system", "user"}
            or not isinstance(content, str)
            or role in messages
        ):
            return _unavailable_structured_prompt()
        messages[role] = content
    user_prompt = messages.get("user")
    if not user_prompt:
        return _unavailable_structured_prompt()
    return _PromptProjection(
        messages.get("system"),
        user_prompt,
        PromptFidelity.EXACT_REQUEST,
    )


def _unavailable_structured_prompt() -> _PromptProjection:
    return _PromptProjection(
        None,
        None,
        PromptFidelity.UNAVAILABLE,
        ("structured_prompt_unavailable",),
    )


def _request_controls(
    request_json: Mapping[str, JsonValue],
) -> _RequestControls:
    config = request_json.get("llm_config")
    if not isinstance(config, Mapping):
        return _RequestControls()
    reasoning = config.get("reasoning")
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        sampling = {}
    return _RequestControls(
        provider=_optional_string(config.get("provider")),
        model=_optional_string(config.get("model")),
        reasoning_json=(
            canonical_json(cast(JsonValue, reasoning))
            if isinstance(reasoning, Mapping)
            else None
        ),
        temperature=_optional_float(sampling.get("temperature")),
        top_p=_optional_float(sampling.get("top_p")),
        max_tokens=_optional_int(config.get("max_tokens")),
    )


def _response_model(
    response_json: Mapping[str, JsonValue] | None,
    controls: _RequestControls,
) -> tuple[str | None, str | None]:
    response_json = response_json or {}
    return (
        _optional_string(response_json.get("provider")) or controls.provider,
        _optional_string(response_json.get("model")) or controls.model,
    )


def _budget_value(
    prompt_template_id: str | None, user_prompt: str | None
) -> int | None:
    template_value: int | None = None
    prompt_value: int | None = None
    if prompt_template_id is not None:
        if match := _BUDGET_ID_RE.search(prompt_template_id):
            template_value = int(match.group("value"))
    if user_prompt is not None:
        if match := _BUDGET_PROMPT_RE.search(user_prompt):
            prompt_value = int(match.group("value"))
    if (
        template_value is not None
        and prompt_value is not None
        and template_value != prompt_value
    ):
        raise ValueError(
            "encoder prompt budget disagrees with template: "
            f"prompt={prompt_value}, template={template_value}, "
            f"template_id={prompt_template_id!r}"
        )
    return template_value if template_value is not None else prompt_value


def _aggregate_fidelity(
    encoder: _PromptProjection | None, decoder: _PromptProjection
) -> PromptFidelity:
    fidelities = [decoder.fidelity]
    if encoder is not None:
        fidelities.append(encoder.fidelity)
    if PromptFidelity.UNAVAILABLE in fidelities:
        return PromptFidelity.UNAVAILABLE
    if PromptFidelity.RECOVERED_TASK in fidelities:
        return PromptFidelity.RECOVERED_TASK
    return PromptFidelity.EXACT_REQUEST


def _task_prompt(task: TaskRecord) -> str:
    import json

    payload = json.loads(task.task_json)
    prompt = payload.get("prompt") if isinstance(payload, Mapping) else None
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"task record {task.task_record_id} has no prompt")
    return prompt


def _require_task(tasks: HumanEvalTaskAdapter, identity: _Id) -> TaskRecord:
    task = tasks.resolve(identity.data_sample_id)
    if task is None or task.task_id != identity.task_id:
        raise ValueError(
            f"HumanEval identity does not resolve to task material: "
            f"{identity.data_sample_id!r}"
        )
    return task


def _response_text(
    response_json: Mapping[str, JsonValue] | None,
) -> str | None:
    if response_json is None:
        return None
    value = response_json.get("text")
    return value if isinstance(value, str) else None


def _response_finish_reason(
    response_json: Mapping[str, JsonValue] | None,
) -> str | None:
    if response_json is None:
        return None
    return _optional_string(response_json.get("finish_reason"))


def _lifecycle(
    stage: Stage, *, output: str | None, status: str | None
) -> LifecycleState:
    if _is_nonblank(output):
        return (
            LifecycleState.PENDING_VALIDATION
            if stage is Stage.DECODER
            else LifecycleState.ENCODER_ONLY
        )
    if status is not None and status.casefold() == "failed":
        return LifecycleState.FAILED
    return LifecycleState.SEEDED


def _is_partial(
    finish_reason: str | None, response_finish_reason: str | None
) -> bool:
    effective = response_finish_reason or finish_reason
    return effective is None or effective.casefold() != "stop"


def _status(row: DumpedPoolRow) -> str | None:
    values: list[tuple[str, object]] = [
        ("metadata_json.status", row.metadata_json.get("status")),
        ("key_values.status", row.key_values.get("status")),
    ]
    present: list[tuple[str, str]] = []
    for location, value in values:
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"row {source_record_id(row)} has invalid {location}"
            )
        present.append((location, value))
    if len({value for _, value in present}) > 1:
        raise ValueError(
            f"row {source_record_id(row)} has conflicting status values: "
            f"{present!r}"
        )
    return present[0][1] if present else None


def _warnings(values: tuple[str, ...]) -> str | None:
    return ";".join(sorted(set(values))) or None


def _nested_string(value: object, key: str) -> str | None:
    return (
        _optional_string(value.get(key))
        if isinstance(value, Mapping)
        else None
    )


def _first_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _is_nonblank(value: str | None) -> bool:
    return value is not None and bool(value.strip())


__all__ = ["HumanEvalAdapter"]
