#!/usr/bin/env python3

"""Reconstruct a generation corpus from historical dr-llm pool dumps."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum, UNIQUE, verify
from pathlib import Path
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_ROOT: Final = Path(__file__).parents[1]
_DEFAULT_SNAPSHOT: Final = (
    _ROOT / "tests" / "corpus" / "humanevalplus_snapshot.json"
)
_SOURCE_MANIFEST_NAME: Final = "manifest.json"
_CORPUS_NAME: Final = "legacy-humaneval-generation-corpus.parquet"
_REQUESTS_NAME: Final = "legacy-humaneval-generation-requests.parquet"
_MANIFEST_NAME: Final = "legacy-humaneval-generation-corpus.manifest.json"
_SOURCE_KIND: Final = "legacy_dr_llm_pool_attempt"
_SOURCE_SCHEMA: Final = "pool"
_DATE_KIND: Final = "created_at"
_ADAPTER_VERSION: Final = 1
_TASK_ID_RE: Final = re.compile(r"^HumanEval/(?P<index>\d+)$")
_BUDGET_ID_RE: Final = re.compile(r"(?:^|/)var_budget=(?P<value>\d+)(?:/|$)")
_BUDGET_PROMPT_RE: Final = re.compile(
    r"\busing at most (?P<value>\d+) characters?\b", re.IGNORECASE
)
_ENCODER_REFERENCE_RE: Final = re.compile(
    r"^encoder_pool/(?P<pool>[^/]+)/(?P<sample_id>[^/]+)$"
)
_CONTENT_COLUMNS: Final = (
    "encoder_system_prompt",
    "encoder_user_prompt",
    "encoder_output",
    "decoder_system_prompt",
    "decoder_user_prompt",
    "decoder_output",
)


@verify(UNIQUE)
class GenerationMode(StrEnum):
    """Recoverable generation topology for one historical attempt."""

    DIRECT = "direct"
    ENCODER_DECODER = "enc_dec"
    UNRESOLVED_ENCODER = "unresolved_encoder"


@verify(UNIQUE)
class BudgetMode(StrEnum):
    """Recoverable character-budget state for one historical attempt."""

    NO_BUDGET = "no_budget"
    BUDGET = "budget"
    UNRESOLVED = "unresolved"


@verify(UNIQUE)
class PromptFidelity(StrEnum):
    """Relationship between canonical prompt fields and source requests."""

    EXACT_REQUEST = "exact_request"
    SEMANTIC_ONLY = "semantic_only"
    UNAVAILABLE = "unavailable"


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class PoolSchemaColumn(_BoundaryModel):
    name: str


class PoolSchema(_BoundaryModel):
    key_columns: tuple[PoolSchemaColumn, ...] = ()


class PoolManifestEntry(_BoundaryModel):
    project_name: str
    pool_name: str
    file_name: str
    row_count: int
    dumped_row_count: int
    pool_schema_json: PoolSchema


class SourceManifest(_BoundaryModel):
    version: int
    created_at: datetime
    pools: tuple[PoolManifestEntry, ...]


class DumpHints(_BoundaryModel):
    human_eval_task_id: str | None = None
    human_eval_pro_task_id: str | None = None
    output_kind: str
    output_json_path: str | None = None


class DumpedPoolRow(_BoundaryModel):
    project_name: str
    pool_name: str
    sample_id: str
    key_values: dict[str, Any] = Field(default_factory=dict)
    sample_idx: int | None = None
    run_id: str | None = None
    request_json: dict[str, Any] = Field(default_factory=dict)
    response_json: dict[str, Any] | None = None
    finish_reason: str | None = None
    attempt_count: int = 0
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    hints: DumpHints


class EncoderPayload(_BoundaryModel):
    project_name: str
    pool_name: str
    sample_id: str
    key_values: dict[str, Any]
    request_json: dict[str, Any]
    response_json: dict[str, Any] | None


class OutputArtifact(_BoundaryModel):
    path: str
    sha256: str
    rows: int


class BuildManifest(_BoundaryModel):
    format: str = "legacy-humaneval-generation-corpus-v1"
    adapter_version: int
    created_at: datetime
    source_manifest: str
    source_manifest_sha256: str
    source_dump_created_at: datetime
    source_dump_pool_count: int
    decoder_pool_count: int
    corpus: OutputArtifact
    requests: OutputArtifact
    task_count: int
    missing_humaneval_task_ids: tuple[str, ...]
    generation_mode_counts: dict[str, int]
    budget_mode_counts: dict[str, int]
    prompt_fidelity_counts: dict[str, int]
    source_kind_counts: dict[str, int]
    distinct_decoder_outputs: int
    encoder_reference_count: int
    resolved_encoder_reference_count: int
    canonical_content_sha256: str


@dataclass(frozen=True, slots=True)
class EncoderReference:
    project_name: str
    pool_name: str
    sample_id: str

    @property
    def key(self) -> str:
        return _encoder_key(self.project_name, self.pool_name, self.sample_id)


@dataclass(frozen=True, slots=True)
class PromptProjection:
    system_prompt: str | None
    user_prompt: str | None
    fidelity: PromptFidelity
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RequestControls:
    provider: str | None
    model: str | None
    reasoning_json: str | None
    temperature: float | None
    top_p: float | None
    max_tokens: int | None


@dataclass(frozen=True, slots=True)
class BuildStats:
    rows: int
    task_ids: frozenset[str]
    distinct_decoder_outputs: int
    generation_mode_counts: Counter[str]
    budget_mode_counts: Counter[str]
    prompt_fidelity_counts: Counter[str]
    source_kind_counts: Counter[str]
    encoder_reference_count: int
    resolved_encoder_reference_count: int


_CORPUS_SCHEMA: Final = pl.Schema(
    {
        "sample_id": pl.String,
        "source_kind": pl.String,
        "source_database": pl.String,
        "source_schema": pl.String,
        "source_table": pl.String,
        "source_record_id": pl.String,
        "generation_run_id": pl.String,
        "attempt_index": pl.Int64,
        "date": pl.Datetime("us", "UTC"),
        "date_kind": pl.String,
        "task_id": pl.String,
        "model": pl.String,
        "encoder_model": pl.String,
        "decoder_model": pl.String,
        "encoder_system_prompt": pl.String,
        "encoder_user_prompt": pl.String,
        "encoder_output": pl.String,
        "decoder_system_prompt": pl.String,
        "decoder_user_prompt": pl.String,
        "decoder_output": pl.String,
        "is_retry": pl.Boolean,
        "is_partial": pl.Boolean,
        "prompt_fidelity": pl.String,
        "content_sha256": pl.String,
        "extraction_warning": pl.String,
    }
)

_CORPUS_NDJSON_SCHEMA: Final = pl.Schema(
    {
        **{
            name: dtype
            for name, dtype in _CORPUS_SCHEMA.items()
            if name != "date"
        },
        "date": pl.String,
    }
)

_REQUESTS_SCHEMA: Final = pl.Schema(
    {
        "sample_id": pl.String,
        "task_id": pl.String,
        "generation_mode": pl.String,
        "budget_mode": pl.String,
        "max_characters": pl.Int64,
        "source_project": pl.String,
        "source_pool": pl.String,
        "source_sample_id": pl.String,
        "source_kind": pl.String,
        "source_attempt_count": pl.Int64,
        "finish_reason": pl.String,
        "encoder_source_pool": pl.String,
        "encoder_source_sample_id": pl.String,
        "encoder_config_id": pl.String,
        "decoder_config_id": pl.String,
        "encoder_prompt_template_id": pl.String,
        "decoder_prompt_template_id": pl.String,
        "encoder_provider": pl.String,
        "encoder_model": pl.String,
        "decoder_provider": pl.String,
        "decoder_model": pl.String,
        "encoder_request_json": pl.String,
        "decoder_request_json": pl.String,
        "encoder_reasoning_json": pl.String,
        "decoder_reasoning_json": pl.String,
        "encoder_temperature": pl.Float64,
        "decoder_temperature": pl.Float64,
        "encoder_top_p": pl.Float64,
        "decoder_top_p": pl.Float64,
        "encoder_max_tokens": pl.Int64,
        "decoder_max_tokens": pl.Int64,
        "encoder_prompt_fidelity": pl.String,
        "decoder_prompt_fidelity": pl.String,
    }
)


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path}")
    return path.resolve()


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path.resolve()


def _output_directory(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(values: Sequence[str | None]) -> str:
    return _sha256_bytes(_canonical_json(list(values)).encode("utf-8"))


def _sample_id(row: DumpedPoolRow) -> str:
    identity = "\0".join(
        (_SOURCE_KIND, row.project_name, row.pool_name, row.sample_id)
    )
    return _sha256_bytes(identity.encode("utf-8"))


def _source_record_id(row: DumpedPoolRow) -> str:
    return f"{row.project_name}:{row.pool_name}:{row.sample_id}"


def _encoder_key(project_name: str, pool_name: str, sample_id: str) -> str:
    return "\0".join((project_name, pool_name, sample_id))


def _read_source_manifest(path: Path) -> SourceManifest:
    try:
        manifest = SourceManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ValueError(f"invalid source manifest {path}: {exc}") from exc
    mismatches = [
        entry.pool_name
        for entry in manifest.pools
        if entry.row_count != entry.dumped_row_count
    ]
    if mismatches:
        raise ValueError(
            "source manifest contains incomplete pool dumps: "
            + ", ".join(sorted(mismatches))
        )
    return manifest


def _is_decoder_pool(entry: PoolManifestEntry) -> bool:
    lowered = entry.pool_name.lower()
    if "decoder" in lowered or lowered.startswith("dec_"):
        return True
    return any(
        column.name.startswith("dec_")
        for column in entry.pool_schema_json.key_columns
    )


def _iter_pool_rows(path: Path) -> Iterator[DumpedPoolRow]:
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield DumpedPoolRow.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(
                    f"invalid pool row at {path}:{line_number}: {exc}"
                ) from exc


def _decoder_output(row: DumpedPoolRow) -> str | None:
    if (
        row.hints.human_eval_task_id is None
        or row.hints.human_eval_pro_task_id is not None
        or row.hints.output_kind != "code_text"
        or row.hints.output_json_path != "response_json.text"
        or row.response_json is None
    ):
        return None
    value = row.response_json.get("text")
    return value if isinstance(value, str) and value.strip() else None


def _validated_task_id(row: DumpedPoolRow) -> str:
    task_id = row.hints.human_eval_task_id
    if task_id is None or _TASK_ID_RE.fullmatch(task_id) is None:
        raise ValueError(
            f"row {_source_record_id(row)} has invalid HumanEval task ID "
            f"{task_id!r}"
        )
    return task_id


def _source_kind(row: DumpedPoolRow) -> str:
    value = row.metadata_json.get("source_kind")
    return value if isinstance(value, str) and value else "unknown"


def _encoder_reference(row: DumpedPoolRow) -> EncoderReference | None:
    if _source_kind(row) != "encoder_sample":
        return None
    pool_name = row.metadata_json.get("source_pool_name")
    raw_reference = row.metadata_json.get("source_sample_id")
    if not isinstance(pool_name, str) or not pool_name:
        return None
    if not isinstance(raw_reference, str):
        raise ValueError(
            f"row {_source_record_id(row)} names encoder pool {pool_name!r} "
            "without a source sample ID"
        )
    match = _ENCODER_REFERENCE_RE.fullmatch(raw_reference)
    if match is None or match.group("pool") != pool_name:
        raise ValueError(
            f"row {_source_record_id(row)} has malformed encoder reference "
            f"{raw_reference!r} for pool {pool_name!r}"
        )
    return EncoderReference(
        project_name=row.project_name,
        pool_name=pool_name,
        sample_id=match.group("sample_id"),
    )


def _collect_encoder_references(
    dump_directory: Path,
    decoder_pools: Sequence[PoolManifestEntry],
) -> tuple[frozenset[EncoderReference], int]:
    references: set[EncoderReference] = set()
    decoder_rows = 0
    for entry in decoder_pools:
        path = dump_directory / entry.file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        pool_rows = 0
        for row in _iter_pool_rows(path):
            pool_rows += 1
            if _decoder_output(row) is None:
                continue
            _validated_task_id(row)
            decoder_rows += 1
            if reference := _encoder_reference(row):
                references.add(reference)
        if pool_rows != entry.dumped_row_count:
            raise ValueError(
                f"pool dump row count mismatch for {entry.pool_name}: "
                f"manifest={entry.dumped_row_count:,}, file={pool_rows:,}"
            )
    if decoder_rows == 0:
        raise ValueError("source dumps contain no HumanEval decoder attempts")
    print(
        f"Located {decoder_rows:,} decoder attempts and "
        f"{len(references):,} encoder references"
    )
    return frozenset(references), decoder_rows


def _create_encoder_store(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE encoders (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    return connection


def _populate_encoder_store(
    connection: sqlite3.Connection,
    *,
    dump_directory: Path,
    manifest: SourceManifest,
    references: frozenset[EncoderReference],
) -> int:
    references_by_pool: dict[tuple[str, str], set[str]] = defaultdict(set)
    for reference in references:
        references_by_pool[(reference.project_name, reference.pool_name)].add(
            reference.sample_id
        )

    entries = {
        (entry.project_name, entry.pool_name): entry
        for entry in manifest.pools
    }
    missing_pools = sorted(set(references_by_pool).difference(entries))
    if missing_pools:
        raise ValueError(f"encoder source pools are absent: {missing_pools}")

    inserted = 0
    for pool_identity, sample_ids in sorted(references_by_pool.items()):
        entry = entries[pool_identity]
        path = dump_directory / entry.file_name
        pool_inserts: list[tuple[str, str]] = []
        pool_rows = 0
        for row in _iter_pool_rows(path):
            pool_rows += 1
            if row.sample_id not in sample_ids:
                continue
            payload = EncoderPayload(
                project_name=row.project_name,
                pool_name=row.pool_name,
                sample_id=row.sample_id,
                key_values=row.key_values,
                request_json=row.request_json,
                response_json=row.response_json,
            )
            pool_inserts.append(
                (
                    _encoder_key(
                        row.project_name, row.pool_name, row.sample_id
                    ),
                    payload.model_dump_json(),
                )
            )
            if len(pool_inserts) >= 1_000:
                connection.executemany(
                    "INSERT INTO encoders(identity, payload) VALUES (?, ?)",
                    pool_inserts,
                )
                inserted += len(pool_inserts)
                pool_inserts.clear()
        if pool_inserts:
            connection.executemany(
                "INSERT INTO encoders(identity, payload) VALUES (?, ?)",
                pool_inserts,
            )
            inserted += len(pool_inserts)
        if pool_rows != entry.dumped_row_count:
            raise ValueError(
                f"pool dump row count mismatch for {entry.pool_name}: "
                f"manifest={entry.dumped_row_count:,}, file={pool_rows:,}"
            )
        connection.commit()

    if inserted != len(references):
        found = {
            row[0]
            for row in connection.execute("SELECT identity FROM encoders")
        }
        missing = sorted(
            reference.key
            for reference in references
            if reference.key not in found
        )
        raise ValueError(
            f"resolved {inserted:,}/{len(references):,} encoder references; "
            f"first missing identities: {missing[:5]}"
        )
    print(f"Resolved all {inserted:,} encoder references")
    return inserted


def _load_encoder(
    connection: sqlite3.Connection, reference: EncoderReference
) -> EncoderPayload:
    result = connection.execute(
        "SELECT payload FROM encoders WHERE identity = ?", (reference.key,)
    ).fetchone()
    if result is None:
        raise RuntimeError(
            f"encoder reference was not indexed: {reference.key}"
        )
    return EncoderPayload.model_validate_json(result[0])


def _prompt_projection(
    request_json: Mapping[str, Any],
    *,
    fallback_user_prompt: str | None = None,
) -> PromptProjection:
    if request_json.get("unavailable") is True:
        if fallback_user_prompt is None:
            return PromptProjection(
                system_prompt=None,
                user_prompt=None,
                fidelity=PromptFidelity.UNAVAILABLE,
                warnings=("original_request_unavailable",),
            )
        return PromptProjection(
            system_prompt=None,
            user_prompt=fallback_user_prompt,
            fidelity=PromptFidelity.SEMANTIC_ONLY,
            warnings=("original_request_unavailable",),
        )

    prompt = request_json.get("prompt")
    if isinstance(prompt, str) and prompt:
        return PromptProjection(
            system_prompt=None,
            user_prompt=prompt,
            fidelity=PromptFidelity.EXACT_REQUEST,
        )
    if not isinstance(prompt, list):
        return PromptProjection(
            system_prompt=None,
            user_prompt=fallback_user_prompt,
            fidelity=(
                PromptFidelity.SEMANTIC_ONLY
                if fallback_user_prompt is not None
                else PromptFidelity.UNAVAILABLE
            ),
            warnings=("structured_prompt_unavailable",),
        )

    messages: dict[str, list[str]] = defaultdict(list)
    unsupported_roles: set[str] = set()
    for item in prompt:
        if not isinstance(item, Mapping):
            unsupported_roles.add(type(item).__name__)
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            unsupported_roles.add(str(role))
            continue
        if role not in {"system", "user"}:
            unsupported_roles.add(role)
            continue
        messages[role].append(content)

    warnings: list[str] = []
    if unsupported_roles:
        warnings.append("unsupported_prompt_roles")
    if len(messages["system"]) > 1 or len(messages["user"]) > 1:
        warnings.append("multiple_prompt_messages_flattened")
    system_prompt = "\n\n".join(messages["system"]) or None
    user_prompt = "\n\n".join(messages["user"]) or fallback_user_prompt
    if user_prompt is None:
        warnings.append("user_prompt_unavailable")
    fidelity = (
        PromptFidelity.EXACT_REQUEST
        if not warnings and user_prompt is not None
        else (
            PromptFidelity.SEMANTIC_ONLY
            if user_prompt is not None
            else PromptFidelity.UNAVAILABLE
        )
    )
    return PromptProjection(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fidelity=fidelity,
        warnings=tuple(sorted(set(warnings))),
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _request_controls(request_json: Mapping[str, Any]) -> RequestControls:
    config = request_json.get("llm_config")
    if not isinstance(config, Mapping):
        return RequestControls(None, None, None, None, None, None)
    reasoning = config.get("reasoning")
    sampling = config.get("sampling")
    if not isinstance(sampling, Mapping):
        sampling = {}
    return RequestControls(
        provider=_optional_string(config.get("provider")),
        model=_optional_string(config.get("model")),
        reasoning_json=(
            _canonical_json(reasoning)
            if isinstance(reasoning, Mapping)
            else None
        ),
        temperature=_optional_float(sampling.get("temperature")),
        top_p=_optional_float(sampling.get("top_p")),
        max_tokens=_optional_int(config.get("max_tokens")),
    )


def _response_model(
    response_json: Mapping[str, Any] | None,
    controls: RequestControls,
) -> tuple[str | None, str | None]:
    response_json = response_json or {}
    provider = (
        _optional_string(response_json.get("provider")) or controls.provider
    )
    model = _optional_string(response_json.get("model")) or controls.model
    return provider, model


def _canonical_model(provider: str | None, model: str | None) -> str | None:
    if model is None:
        return None
    if provider == "openrouter" and "/" in model:
        return model
    if provider is None or model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


def _response_text(response_json: Mapping[str, Any] | None) -> str | None:
    if response_json is None:
        return None
    return _optional_string(response_json.get("text"))


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _budget_value(
    prompt_template_id: str | None, user_prompt: str | None
) -> int | None:
    template_value: int | None = None
    prompt_value: int | None = None
    if prompt_template_id is not None:
        match = _BUDGET_ID_RE.search(prompt_template_id)
        if match:
            template_value = int(match.group("value"))
    if user_prompt is not None:
        match = _BUDGET_PROMPT_RE.search(user_prompt)
        if match:
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
    encoder: PromptProjection | None, decoder: PromptProjection
) -> PromptFidelity:
    fidelities = [decoder.fidelity]
    if encoder is not None:
        fidelities.append(encoder.fidelity)
    if PromptFidelity.UNAVAILABLE in fidelities:
        return PromptFidelity.UNAVAILABLE
    if PromptFidelity.SEMANTIC_ONLY in fidelities:
        return PromptFidelity.SEMANTIC_ONLY
    return PromptFidelity.EXACT_REQUEST


def _validate_encoder_lineage(
    row: DumpedPoolRow,
    encoder: EncoderPayload,
    encoder_output: str,
    encoder_config_id: str | None,
    encoder_prompt_template_id: str | None,
) -> None:
    embedded_output = _first_string(
        row.metadata_json.get("source_text"),
        (row.metadata_json.get("source_sample_payload") or {}).get("text")
        if isinstance(row.metadata_json.get("source_sample_payload"), Mapping)
        else None,
    )
    if embedded_output is not None and embedded_output != encoder_output:
        raise ValueError(
            f"row {_source_record_id(row)} embeds encoder output that does not "
            "match its referenced encoder row"
        )
    expected_config = _optional_string(
        row.metadata_json.get("enc_llm_config_id")
    )
    if expected_config is not None and expected_config != encoder_config_id:
        raise ValueError(
            f"row {_source_record_id(row)} encoder config mismatch: "
            f"{expected_config!r} != {encoder_config_id!r}"
        )
    expected_prompt = _optional_string(
        row.metadata_json.get("enc_prompt_template_id")
    )
    if (
        expected_prompt is not None
        and expected_prompt != encoder_prompt_template_id
    ):
        raise ValueError(
            f"row {_source_record_id(row)} encoder prompt mismatch: "
            f"{expected_prompt!r} != {encoder_prompt_template_id!r}"
        )


def _build_output_rows(
    row: DumpedPoolRow,
    *,
    encoder_store: sqlite3.Connection,
    prompts_by_task_id: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    decoder_output = _decoder_output(row)
    if decoder_output is None:
        raise ValueError("attempt builder received a non-decoder row")
    task_id = _validated_task_id(row)
    if row.created_at is None:
        raise ValueError(f"row {_source_record_id(row)} has no creation date")

    source_kind = _source_kind(row)
    decoder_projection = _prompt_projection(
        row.request_json,
        fallback_user_prompt=prompts_by_task_id.get(task_id),
    )
    decoder_controls = _request_controls(row.request_json)
    decoder_provider, decoder_model_raw = _response_model(
        row.response_json, decoder_controls
    )
    decoder_model = _canonical_model(decoder_provider, decoder_model_raw)
    if decoder_model is None:
        raise ValueError(f"row {_source_record_id(row)} has no decoder model")

    warnings = set(decoder_projection.warnings)
    reference = _encoder_reference(row)
    encoder_projection: PromptProjection | None = None
    encoder_controls = RequestControls(None, None, None, None, None, None)
    encoder_provider: str | None = None
    encoder_model_raw: str | None = None
    encoder_model: str | None = None
    encoder_output: str | None = None
    encoder_request_json: str | None = None
    encoder_config_id: str | None = None
    encoder_prompt_template_id: str | None = None

    if reference is not None:
        encoder = _load_encoder(encoder_store, reference)
        encoder_projection = _prompt_projection(encoder.request_json)
        encoder_controls = _request_controls(encoder.request_json)
        encoder_provider, encoder_model_raw = _response_model(
            encoder.response_json, encoder_controls
        )
        encoder_model = _canonical_model(encoder_provider, encoder_model_raw)
        encoder_output = _response_text(encoder.response_json)
        encoder_config_id = _first_string(
            encoder.key_values.get("llm_config_id"),
            row.metadata_json.get("enc_llm_config_id"),
        )
        encoder_prompt_template_id = _first_string(
            encoder.key_values.get("prompt_template_id"),
            row.metadata_json.get("enc_prompt_template_id"),
        )
        if encoder_model is None or encoder_output is None:
            raise ValueError(
                f"encoder {reference.key} has no model or nonblank output"
            )
        _validate_encoder_lineage(
            row,
            encoder,
            encoder_output,
            encoder_config_id,
            encoder_prompt_template_id,
        )
        warnings.update(encoder_projection.warnings)
        generation_mode = GenerationMode.ENCODER_DECODER
        encoder_request_json = _canonical_json(encoder.request_json)
        maximum = _budget_value(
            encoder_prompt_template_id, encoder_projection.user_prompt
        )
        budget_mode = (
            BudgetMode.BUDGET if maximum is not None else BudgetMode.NO_BUDGET
        )
    elif source_kind == "encoder_sample":
        generation_mode = GenerationMode.UNRESOLVED_ENCODER
        budget_mode = BudgetMode.UNRESOLVED
        maximum = None
        warnings.add("encoder_source_unresolved")
    else:
        generation_mode = GenerationMode.DIRECT
        budget_mode = BudgetMode.NO_BUDGET
        maximum = None

    decoder_config_id = _first_string(
        row.key_values.get("dec_llm_config_id"),
        row.key_values.get("llm_config_id"),
    )
    decoder_prompt_template_id = _first_string(
        row.key_values.get("dec_prompt_template_id"),
        row.key_values.get("prompt_template_id"),
    )
    fidelity = _aggregate_fidelity(encoder_projection, decoder_projection)
    sample_id = _sample_id(row)
    source_record_id = _source_record_id(row)
    content_values = (
        encoder_projection.system_prompt if encoder_projection else None,
        encoder_projection.user_prompt if encoder_projection else None,
        encoder_output,
        decoder_projection.system_prompt,
        decoder_projection.user_prompt,
        decoder_output,
    )

    canonical_row = {
        "sample_id": sample_id,
        "source_kind": _SOURCE_KIND,
        "source_database": row.project_name,
        "source_schema": _SOURCE_SCHEMA,
        "source_table": row.pool_name,
        "source_record_id": source_record_id,
        "generation_run_id": row.run_id or source_record_id,
        "attempt_index": (
            row.attempt_count - 1 if row.attempt_count > 0 else None
        ),
        "date": row.created_at.astimezone(UTC).isoformat(),
        "date_kind": _DATE_KIND,
        "task_id": task_id,
        "model": decoder_model,
        "encoder_model": encoder_model,
        "decoder_model": decoder_model,
        "encoder_system_prompt": (
            encoder_projection.system_prompt if encoder_projection else None
        ),
        "encoder_user_prompt": (
            encoder_projection.user_prompt if encoder_projection else None
        ),
        "encoder_output": encoder_output,
        "decoder_system_prompt": decoder_projection.system_prompt,
        "decoder_user_prompt": decoder_projection.user_prompt,
        "decoder_output": decoder_output,
        "is_retry": row.attempt_count > 1,
        "is_partial": row.finish_reason == "length",
        "prompt_fidelity": fidelity.value,
        "content_sha256": _content_sha256(content_values),
        "extraction_warning": ";".join(sorted(warnings)) or None,
    }
    request_row = {
        "sample_id": sample_id,
        "task_id": task_id,
        "generation_mode": generation_mode.value,
        "budget_mode": budget_mode.value,
        "max_characters": maximum,
        "source_project": row.project_name,
        "source_pool": row.pool_name,
        "source_sample_id": row.sample_id,
        "source_kind": source_kind,
        "source_attempt_count": row.attempt_count,
        "finish_reason": row.finish_reason,
        "encoder_source_pool": reference.pool_name if reference else None,
        "encoder_source_sample_id": reference.sample_id if reference else None,
        "encoder_config_id": encoder_config_id,
        "decoder_config_id": decoder_config_id,
        "encoder_prompt_template_id": encoder_prompt_template_id,
        "decoder_prompt_template_id": decoder_prompt_template_id,
        "encoder_provider": encoder_provider,
        "encoder_model": encoder_model_raw,
        "decoder_provider": decoder_provider,
        "decoder_model": decoder_model_raw,
        "encoder_request_json": encoder_request_json,
        "decoder_request_json": _canonical_json(row.request_json),
        "encoder_reasoning_json": encoder_controls.reasoning_json,
        "decoder_reasoning_json": decoder_controls.reasoning_json,
        "encoder_temperature": encoder_controls.temperature,
        "decoder_temperature": decoder_controls.temperature,
        "encoder_top_p": encoder_controls.top_p,
        "decoder_top_p": decoder_controls.top_p,
        "encoder_max_tokens": encoder_controls.max_tokens,
        "decoder_max_tokens": decoder_controls.max_tokens,
        "encoder_prompt_fidelity": (
            encoder_projection.fidelity.value if encoder_projection else None
        ),
        "decoder_prompt_fidelity": decoder_projection.fidelity.value,
    }
    return canonical_row, request_row


def _load_humaneval_prompts(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no HumanEval rows")
    prompts: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        if isinstance(task_id, str) and isinstance(prompt, str) and prompt:
            prompts[task_id] = prompt
    if len(prompts) != 164:
        raise ValueError(
            f"expected 164 HumanEval prompts, found {len(prompts)}"
        )
    return prompts


def _write_ndjson_row(file: Any, row: Mapping[str, Any]) -> None:
    file.write(_canonical_json(row))
    file.write("\n")


def _scan_decoder_rows(
    *,
    dump_directory: Path,
    decoder_pools: Sequence[PoolManifestEntry],
    encoder_store: sqlite3.Connection,
    prompts_by_task_id: Mapping[str, str],
    corpus_ndjson: Path,
    requests_ndjson: Path,
    expected_rows: int,
    encoder_reference_count: int,
    resolved_encoder_reference_count: int,
) -> BuildStats:
    task_ids: set[str] = set()
    decoder_outputs: set[str] = set()
    sample_ids: set[str] = set()
    generation_modes: Counter[str] = Counter()
    budget_modes: Counter[str] = Counter()
    prompt_fidelities: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    rows = 0

    with (
        corpus_ndjson.open("w", encoding="utf-8") as corpus_file,
        requests_ndjson.open("w", encoding="utf-8") as requests_file,
    ):
        for entry in decoder_pools:
            path = dump_directory / entry.file_name
            pool_rows = 0
            for row in _iter_pool_rows(path):
                pool_rows += 1
                output = _decoder_output(row)
                if output is None:
                    continue
                corpus_row, request_row = _build_output_rows(
                    row,
                    encoder_store=encoder_store,
                    prompts_by_task_id=prompts_by_task_id,
                )
                sample_id = str(corpus_row["sample_id"])
                if sample_id in sample_ids:
                    raise ValueError(
                        f"duplicate generated sample ID {sample_id}"
                    )
                sample_ids.add(sample_id)
                _write_ndjson_row(corpus_file, corpus_row)
                _write_ndjson_row(requests_file, request_row)
                rows += 1
                task_ids.add(str(corpus_row["task_id"]))
                decoder_outputs.add(output)
                generation_modes[str(request_row["generation_mode"])] += 1
                budget_modes[str(request_row["budget_mode"])] += 1
                prompt_fidelities[str(corpus_row["prompt_fidelity"])] += 1
                source_kinds[str(request_row["source_kind"])] += 1
                if rows % 25_000 == 0:
                    print(f"Reconstructed {rows:,} generation rows")
            if pool_rows != entry.dumped_row_count:
                raise ValueError(
                    f"pool dump row count mismatch for {entry.pool_name}: "
                    f"manifest={entry.dumped_row_count:,}, file={pool_rows:,}"
                )

    if rows != expected_rows:
        raise ValueError(
            f"decoder row count changed between passes: {expected_rows:,} != {rows:,}"
        )
    return BuildStats(
        rows=rows,
        task_ids=frozenset(task_ids),
        distinct_decoder_outputs=len(decoder_outputs),
        generation_mode_counts=generation_modes,
        budget_mode_counts=budget_modes,
        prompt_fidelity_counts=prompt_fidelities,
        source_kind_counts=source_kinds,
        encoder_reference_count=encoder_reference_count,
        resolved_encoder_reference_count=resolved_encoder_reference_count,
    )


def _ndjson_to_parquet(
    ndjson: Path, output: Path, schema: pl.Schema, *, date_column: bool = False
) -> None:
    lazy = pl.scan_ndjson(ndjson, schema=schema)
    if date_column:
        lazy = lazy.with_columns(
            pl.col("date").str.to_datetime(time_zone="UTC", strict=True)
        ).select(_CORPUS_SCHEMA.names())
    lazy.sink_parquet(output, compression="zstd", maintain_order=True)


def _validate_schema(path: Path, expected: pl.Schema) -> None:
    actual = pl.read_parquet_schema(path)
    if actual != expected:
        raise ValueError(
            f"schema mismatch for {path}:\nexpected={expected}\nactual={actual}"
        )


def _validate_outputs(
    corpus_path: Path,
    requests_path: Path,
    *,
    stats: BuildStats,
) -> None:
    _validate_schema(corpus_path, _CORPUS_SCHEMA)
    _validate_schema(requests_path, _REQUESTS_SCHEMA)
    corpus = pl.scan_parquet(corpus_path)
    requests = pl.scan_parquet(requests_path)
    corpus_checks = (
        corpus.select(
            pl.len().alias("rows"),
            pl.col("sample_id").n_unique().alias("sample_ids"),
            pl.col("task_id").n_unique().alias("tasks"),
            pl.col("decoder_output").is_null().sum().alias("null_outputs"),
            pl.col("decoder_model").is_null().sum().alias("null_models"),
        )
        .collect()
        .row(0, named=True)
    )
    request_checks = (
        requests.select(
            pl.len().alias("rows"),
            pl.col("sample_id").n_unique().alias("sample_ids"),
        )
        .collect()
        .row(0, named=True)
    )
    if corpus_checks != {
        "rows": stats.rows,
        "sample_ids": stats.rows,
        "tasks": len(stats.task_ids),
        "null_outputs": 0,
        "null_models": 0,
    }:
        raise ValueError(
            f"canonical corpus validation failed: {corpus_checks}"
        )
    if request_checks != {"rows": stats.rows, "sample_ids": stats.rows}:
        raise ValueError(
            f"request sidecar validation failed: {request_checks}"
        )

    unmatched = (
        corpus.select("sample_id")
        .join(requests.select("sample_id"), on="sample_id", how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    if unmatched:
        raise ValueError(
            f"{unmatched} corpus rows have no request sidecar row"
        )

    joined = corpus.select(
        "sample_id", "encoder_model", "encoder_output"
    ).join(
        requests.select("sample_id", "generation_mode"),
        on="sample_id",
        validate="1:1",
    )
    incomplete_enc_dec = (
        joined.filter(
            (pl.col("generation_mode") == GenerationMode.ENCODER_DECODER.value)
            & (
                pl.col("encoder_model").is_null()
                | pl.col("encoder_output").is_null()
            )
        )
        .select(pl.len())
        .collect()
        .item()
    )
    if incomplete_enc_dec:
        raise ValueError(
            f"{incomplete_enc_dec} enc-dec rows lack encoder model/output"
        )

    content = pl.read_parquet(
        corpus_path, columns=list(_CONTENT_COLUMNS) + ["content_sha256"]
    )
    for row in content.iter_rows(named=True):
        expected = _content_sha256(
            [row[column] for column in _CONTENT_COLUMNS]
        )
        if row["content_sha256"] != expected:
            raise ValueError("content hash validation failed")
    print(
        f"Validated {stats.rows:,} canonical rows, request sidecars, schemas, "
        "encoder completeness, and content hashes"
    )


def _missing_task_ids(task_ids: Iterable[str]) -> tuple[str, ...]:
    observed = set(task_ids)
    return tuple(
        task_id
        for task_id in (f"HumanEval/{index}" for index in range(164))
        if task_id not in observed
    )


def _manifest(
    *,
    source_manifest_path: Path,
    source_manifest: SourceManifest,
    decoder_pool_count: int,
    corpus_path: Path,
    requests_path: Path,
    stats: BuildStats,
) -> BuildManifest:
    return BuildManifest(
        adapter_version=_ADAPTER_VERSION,
        created_at=datetime.now(UTC),
        source_manifest=str(source_manifest_path),
        source_manifest_sha256=_sha256_file(source_manifest_path),
        source_dump_created_at=source_manifest.created_at,
        source_dump_pool_count=len(source_manifest.pools),
        decoder_pool_count=decoder_pool_count,
        corpus=OutputArtifact(
            path=_CORPUS_NAME,
            sha256=_sha256_file(corpus_path),
            rows=stats.rows,
        ),
        requests=OutputArtifact(
            path=_REQUESTS_NAME,
            sha256=_sha256_file(requests_path),
            rows=stats.rows,
        ),
        task_count=len(stats.task_ids),
        missing_humaneval_task_ids=_missing_task_ids(stats.task_ids),
        generation_mode_counts=dict(
            sorted(stats.generation_mode_counts.items())
        ),
        budget_mode_counts=dict(sorted(stats.budget_mode_counts.items())),
        prompt_fidelity_counts=dict(
            sorted(stats.prompt_fidelity_counts.items())
        ),
        source_kind_counts=dict(sorted(stats.source_kind_counts.items())),
        distinct_decoder_outputs=stats.distinct_decoder_outputs,
        encoder_reference_count=stats.encoder_reference_count,
        resolved_encoder_reference_count=stats.resolved_encoder_reference_count,
        canonical_content_sha256=(
            "sha256(utf8(json.dumps([encoder_system_prompt,"
            "encoder_user_prompt,encoder_output,decoder_system_prompt,"
            "decoder_user_prompt,decoder_output],ensure_ascii=False,"
            "separators=(',',':'))))"
        ),
    )


def _validate_output_destination(output_directory: Path) -> None:
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to write into non-empty output directory {output_directory}"
        )


def build_corpus(
    dump_directory: Path,
    *,
    snapshot_path: Path,
    output_directory: Path,
) -> BuildManifest:
    _validate_output_destination(output_directory)
    source_manifest_path = dump_directory / _SOURCE_MANIFEST_NAME
    source_manifest = _read_source_manifest(source_manifest_path)
    decoder_pools = tuple(
        entry for entry in source_manifest.pools if _is_decoder_pool(entry)
    )
    if not decoder_pools:
        raise ValueError("source manifest contains no decoder pools")
    prompts_by_task_id = _load_humaneval_prompts(snapshot_path)

    references, expected_rows = _collect_encoder_references(
        dump_directory, decoder_pools
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="legacy-humaneval-corpus-", dir=output_directory.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        encoder_store = _create_encoder_store(temporary / "encoders.sqlite3")
        try:
            resolved_references = _populate_encoder_store(
                encoder_store,
                dump_directory=dump_directory,
                manifest=source_manifest,
                references=references,
            )
            corpus_ndjson = temporary / "corpus.ndjson"
            requests_ndjson = temporary / "requests.ndjson"
            stats = _scan_decoder_rows(
                dump_directory=dump_directory,
                decoder_pools=decoder_pools,
                encoder_store=encoder_store,
                prompts_by_task_id=prompts_by_task_id,
                corpus_ndjson=corpus_ndjson,
                requests_ndjson=requests_ndjson,
                expected_rows=expected_rows,
                encoder_reference_count=len(references),
                resolved_encoder_reference_count=resolved_references,
            )
        finally:
            encoder_store.close()

        corpus_path = temporary / _CORPUS_NAME
        requests_path = temporary / _REQUESTS_NAME
        _ndjson_to_parquet(
            corpus_ndjson,
            corpus_path,
            _CORPUS_NDJSON_SCHEMA,
            date_column=True,
        )
        _ndjson_to_parquet(requests_ndjson, requests_path, _REQUESTS_SCHEMA)
        _validate_outputs(corpus_path, requests_path, stats=stats)
        manifest = _manifest(
            source_manifest_path=source_manifest_path,
            source_manifest=source_manifest,
            decoder_pool_count=len(decoder_pools),
            corpus_path=corpus_path,
            requests_path=requests_path,
            stats=stats,
        )
        manifest_path = temporary / _MANIFEST_NAME
        manifest_path.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        for name in (_CORPUS_NAME, _REQUESTS_NAME, _MANIFEST_NAME):
            os.replace(temporary / name, output_directory / name)

    print(f"Wrote {manifest.corpus.rows:,} rows to {output_directory}")
    print(
        f"tasks={manifest.task_count} "
        f"missing={list(manifest.missing_humaneval_task_ids)} "
        f"generation_modes={manifest.generation_mode_counts} "
        f"budgets={manifest.budget_mode_counts}"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a canonical HumanEval generation corpus and exact "
            "request/config sidecar from historical dr-llm pool dumps."
        )
    )
    parser.add_argument(
        "dump_directory",
        type=_existing_directory,
        help="directory containing the raw per-pool gzip dumps and manifest",
    )
    parser.add_argument(
        "--snapshot",
        type=_existing_file,
        default=_DEFAULT_SNAPSHOT,
        help="HumanEval snapshot used only for unavailable migrated prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=_output_directory,
        required=True,
        help="new or empty destination directory",
    )
    arguments = parser.parse_args(argv)
    try:
        build_corpus(
            arguments.dump_directory,
            snapshot_path=arguments.snapshot,
            output_directory=arguments.output_dir,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
