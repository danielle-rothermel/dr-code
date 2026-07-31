"""Authenticated, deterministic publication for behavioral-mutant datasets."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from dr_exec import EXECUTOR_IDENTITY
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from dr_code.corpus.atomic_directory import staged_output_directory
from dr_code.corpus.stable_files import stable_file
from dr_code.eval.identity import identity_hash_for
from dr_code.mutants.operators import (
    ALL_FAMILIES,
    MutationError,
    OperatorFamily,
    apply_site,
    iter_sites,
)
from dr_code.mutants.outcomes import ExecutionOutcome
from dr_code.mutants.provenance import (
    CanonicalTask,
    canonical_suite_digest,
    current_runtime_identity,
)
from dr_code.synthetic.humaneval_loader import (
    HF_DATASET_ID,
    HF_REVISION,
    HumanEvalSource,
)

DATASET_SCHEMA_VERSION: Final = 2
MANIFEST_SCHEMA_VERSION: Final = 1
GENERATOR_VERSION: Final = "mutants@v2"
RECORDS_FILENAME: Final = "mutants.jsonl"
MANIFEST_FILENAME: Final = "manifest.json"
_CONFIG_SCHEMA: Final = "dr_code.mutants.generation_config"
_RECORD_SCHEMA: Final = "dr_code.mutants.record"
_DATASET_SCHEMA: Final = "dr_code.mutants.dataset"


class _StrictPersistedModel(BaseModel):
    """Strict immutable boundary for mutant JSON artifacts."""

    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class MutantRecord(_StrictPersistedModel):
    """One accepted, execution-validated behavioral mutant."""

    content_identity: str
    task_id: str
    entry_point: str
    prompt: str
    canonical_full_source: str
    mutated_full_source: str
    operator_family: OperatorFamily
    seed: int
    site_node_path: int
    site_target_index: int
    site_description: str
    input_reprs: tuple[str, ...]
    mutant_expected: tuple[ExecutionOutcome, ...]
    canonical_expected: tuple[ExecutionOutcome, ...]
    distinct_input_indices: tuple[int, ...]
    diff_summary: str
    canonical_test: str

    @property
    def distinct_input_count(self) -> int:
        return len(self.distinct_input_indices)


class GenerationConfig(_StrictPersistedModel):
    """Identity-bearing inputs to one generation run."""

    generator_version: Literal["mutants@v2"] = GENERATOR_VERSION
    dataset_schema_version: Literal[2] = DATASET_SCHEMA_VERSION
    dataset_source: HumanEvalSource
    dataset_id: str
    dataset_revision: str
    operator_families: tuple[OperatorFamily, ...]
    seeds: int
    max_inputs_per_mutant: int
    timeout_seconds: float
    task_ids: tuple[str, ...]
    canonical_suite_digest: str
    runner_identity: str
    runtime_identity: str

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _require_json_float(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("timeout_seconds must be a JSON float")
        return value

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=_CONFIG_SCHEMA,
            payload=self.model_dump(mode="json"),
        )


class SkippedMutation(_StrictPersistedModel):
    """One deterministic search coordinate that was not accepted."""

    task_id: str
    operator_family: OperatorFamily | Literal["*"]
    seed: int | None
    reason: str


class FamilyCount(_StrictPersistedModel):
    """Accepted record count for one configured family."""

    operator_family: OperatorFamily
    count: int


class DatasetManifest(_StrictPersistedModel):
    """Manifest authenticating the records artifact and generation config."""

    manifest_schema_version: Literal[1] = MANIFEST_SCHEMA_VERSION
    dataset_schema_version: Literal[2] = DATASET_SCHEMA_VERSION
    generator_version: Literal["mutants@v2"] = GENERATOR_VERSION
    config: GenerationConfig
    config_identity: str
    dataset_identity: str
    records_filename: Literal["mutants.jsonl"] = RECORDS_FILENAME
    records_sha256: str
    accepted_count: int
    accepted_by_family: tuple[FamilyCount, ...]
    skipped: tuple[SkippedMutation, ...]


@dataclass(frozen=True, slots=True)
class GeneratedDataset:
    """Internal generation result awaiting publication."""

    config: GenerationConfig
    canonical_suite: tuple[CanonicalTask, ...]
    records: tuple[MutantRecord, ...]
    accepted_by_family: tuple[FamilyCount, ...]
    skipped: tuple[SkippedMutation, ...]


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """Validated records and their authenticating manifest."""

    records: tuple[MutantRecord, ...]
    manifest: DatasetManifest


@dataclass(frozen=True, slots=True)
class DatasetArtifacts:
    """Paths and manifest from one completed publication."""

    records_path: Path
    manifest_path: Path
    manifest: DatasetManifest


class DatasetValidationError(ValueError):
    """Published mutant artifacts failed validation."""


def build_record(
    *,
    task_id: str,
    entry_point: str,
    prompt: str,
    canonical_full_source: str,
    mutated_full_source: str,
    operator_family: OperatorFamily,
    seed: int,
    site_node_path: int,
    site_target_index: int,
    site_description: str,
    input_reprs: tuple[str, ...],
    mutant_expected: tuple[ExecutionOutcome, ...],
    canonical_expected: tuple[ExecutionOutcome, ...],
    distinct_input_indices: tuple[int, ...],
    diff_summary: str,
    canonical_test: str,
) -> MutantRecord:
    """Build a record with its stable full-content identity."""

    provisional = MutantRecord(
        content_identity="",
        task_id=task_id,
        entry_point=entry_point,
        prompt=prompt,
        canonical_full_source=canonical_full_source,
        mutated_full_source=mutated_full_source,
        operator_family=operator_family,
        seed=seed,
        site_node_path=site_node_path,
        site_target_index=site_target_index,
        site_description=site_description,
        input_reprs=input_reprs,
        mutant_expected=mutant_expected,
        canonical_expected=canonical_expected,
        distinct_input_indices=distinct_input_indices,
        diff_summary=diff_summary,
        canonical_test=canonical_test,
    )
    payload = provisional.model_dump(
        mode="json",
        exclude={"content_identity"},
    )
    content_identity = identity_hash_for(
        schema=_RECORD_SCHEMA,
        payload=payload,
    )
    return provisional.model_copy(
        update={"content_identity": content_identity}
    )


def publish_dataset(
    *,
    output_dir: Path,
    generated: GeneratedDataset,
) -> DatasetArtifacts:
    """Publish one immutable dataset directory without replacing a target."""

    records_bytes = _encode_records(generated.records)
    records_sha256 = _sha256(records_bytes)
    config_identity = generated.config.identity_hash()
    dataset_identity = _dataset_identity(
        config_identity=config_identity,
        records_sha256=records_sha256,
        accepted_count=len(generated.records),
        accepted_by_family=generated.accepted_by_family,
        skipped=generated.skipped,
    )
    manifest = DatasetManifest(
        config=generated.config,
        config_identity=config_identity,
        dataset_identity=dataset_identity,
        records_sha256=records_sha256,
        accepted_count=len(generated.records),
        accepted_by_family=generated.accepted_by_family,
        skipped=generated.skipped,
    )
    _validate_components(
        records=generated.records,
        manifest=manifest,
        records_bytes=records_bytes,
        expected_config_identity=config_identity,
        expected_dataset_identity=dataset_identity,
        canonical_suite=generated.canonical_suite,
        require_current_runtime_identity=True,
    )
    manifest_bytes = _encode_manifest(manifest)
    with staged_output_directory(output_dir) as staged:
        (staged / RECORDS_FILENAME).write_bytes(records_bytes)
        (staged / MANIFEST_FILENAME).write_bytes(manifest_bytes)
    return DatasetArtifacts(
        records_path=output_dir / RECORDS_FILENAME,
        manifest_path=output_dir / MANIFEST_FILENAME,
        manifest=manifest,
    )


def load_dataset(
    output_dir: Path,
    *,
    expected_dataset_identity: str,
    expected_config_identity: str | None = None,
    max_manifest_bytes: int,
    max_records_bytes: int,
) -> LoadedDataset:
    """Load a dataset within caller-trusted manifest and record-size limits.

    The limits bound bytes captured before the manifest or records artifact is
    authenticated.  Callers must derive them from their trusted generation
    policy or storage contract; an artifact's manifest cannot safely supply
    its own resource limits.
    """

    _validate_byte_ceiling(max_manifest_bytes, name="max_manifest_bytes")
    _validate_byte_ceiling(max_records_bytes, name="max_records_bytes")
    if not output_dir.is_dir():
        raise DatasetValidationError(
            f"mutant dataset directory does not exist: {output_dir}"
        )
    manifest_path: Path | None = None
    records_path: Path | None = None
    for entry_count, path in enumerate(output_dir.iterdir(), start=1):
        if entry_count == 3:
            raise DatasetValidationError(
                "mutant dataset directory must contain exactly "
                f"{sorted((MANIFEST_FILENAME, RECORDS_FILENAME))}"
            )
        if path.name == MANIFEST_FILENAME:
            if manifest_path is not None:
                raise DatasetValidationError(
                    "mutant dataset directory must contain exactly "
                    f"{sorted((MANIFEST_FILENAME, RECORDS_FILENAME))}"
                )
            manifest_path = path
        elif path.name == RECORDS_FILENAME:
            if records_path is not None:
                raise DatasetValidationError(
                    "mutant dataset directory must contain exactly "
                    f"{sorted((MANIFEST_FILENAME, RECORDS_FILENAME))}"
                )
            records_path = path
        else:
            raise DatasetValidationError(
                "mutant dataset directory must contain exactly "
                f"{sorted((MANIFEST_FILENAME, RECORDS_FILENAME))}"
            )
    if manifest_path is None or records_path is None:
        raise DatasetValidationError(
            "mutant dataset directory must contain exactly "
            f"{sorted((MANIFEST_FILENAME, RECORDS_FILENAME))}"
        )
    manifest_bytes = _read_bounded_bytes(
        manifest_path,
        label=MANIFEST_FILENAME,
        max_bytes=max_manifest_bytes,
    )
    try:
        manifest = DatasetManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        raise DatasetValidationError("invalid mutant manifest") from exc
    _validate_manifest(
        manifest=manifest,
        expected_config_identity=expected_config_identity,
        expected_dataset_identity=expected_dataset_identity,
        require_current_runtime_identity=False,
    )
    records_bytes = _read_authenticated_records(
        records_path,
        expected_sha256=manifest.records_sha256,
        max_bytes=max_records_bytes,
    )
    records = _decode_records(records_bytes)
    _validate_records(
        records=records,
        manifest=manifest,
        canonical_suite=None,
    )
    return LoadedDataset(records=records, manifest=manifest)


def record_order_key(
    record: MutantRecord,
) -> tuple[tuple[str, int, str], str, int, int, int]:
    """Stable order shared by generation and authenticated loading."""

    return (
        _task_order_key(record.task_id),
        record.operator_family.value,
        record.seed,
        record.site_node_path,
        record.site_target_index,
    )


def skip_order_key(
    skip: SkippedMutation,
    config: GenerationConfig,
) -> tuple[tuple[str, int, str], int, int]:
    """Stable order for the persisted deterministic search log."""

    family_rank = (
        -1
        if skip.operator_family == "*"
        else config.operator_families.index(skip.operator_family)
    )
    seed_rank = -1 if skip.seed is None else skip.seed
    return _task_order_key(skip.task_id), family_rank, seed_rank


def _validate_components(
    *,
    records: tuple[MutantRecord, ...],
    manifest: DatasetManifest,
    records_bytes: bytes,
    expected_config_identity: str | None,
    expected_dataset_identity: str,
    canonical_suite: tuple[CanonicalTask, ...] | None,
    require_current_runtime_identity: bool,
) -> None:
    _validate_manifest(
        manifest=manifest,
        expected_config_identity=expected_config_identity,
        expected_dataset_identity=expected_dataset_identity,
        require_current_runtime_identity=require_current_runtime_identity,
    )
    _validate_records_sha256(
        actual_sha256=_sha256(records_bytes),
        expected_sha256=manifest.records_sha256,
    )
    _validate_records(
        records=records,
        manifest=manifest,
        canonical_suite=canonical_suite,
    )


def _validate_manifest(
    *,
    manifest: DatasetManifest,
    expected_config_identity: str | None,
    expected_dataset_identity: str,
    require_current_runtime_identity: bool,
) -> None:
    if manifest.dataset_identity != expected_dataset_identity:
        raise DatasetValidationError("unexpected mutant dataset identity")
    config_identity = manifest.config.identity_hash()
    if manifest.config_identity != config_identity:
        raise DatasetValidationError("manifest config identity mismatch")
    if (
        expected_config_identity is not None
        and config_identity != expected_config_identity
    ):
        raise DatasetValidationError("unexpected generation config identity")
    recomputed_dataset_identity = _dataset_identity(
        config_identity=config_identity,
        records_sha256=manifest.records_sha256,
        accepted_count=manifest.accepted_count,
        accepted_by_family=manifest.accepted_by_family,
        skipped=manifest.skipped,
    )
    if manifest.dataset_identity != recomputed_dataset_identity:
        raise DatasetValidationError("manifest dataset identity mismatch")

    _validate_config(
        manifest.config,
        require_current_runtime_identity=require_current_runtime_identity,
    )


def _validate_records(
    *,
    records: tuple[MutantRecord, ...],
    manifest: DatasetManifest,
    canonical_suite: tuple[CanonicalTask, ...] | None,
) -> None:
    canonical_by_id: dict[str, CanonicalTask] | None = None
    if canonical_suite is not None:
        if tuple(task.task_id for task in canonical_suite) != (
            manifest.config.task_ids
        ):
            raise DatasetValidationError(
                "captured canonical suite task order mismatch"
            )
        if (
            canonical_suite_digest(canonical_suite)
            != manifest.config.canonical_suite_digest
        ):
            raise DatasetValidationError(
                "generation config canonical suite digest mismatch"
            )
        canonical_by_id = {task.task_id: task for task in canonical_suite}

    if manifest.accepted_count != len(records):
        raise DatasetValidationError("manifest accepted count mismatch")
    if tuple(sorted(records, key=record_order_key)) != records:
        raise DatasetValidationError("mutant records are not in stable order")
    if (
        tuple(
            sorted(
                manifest.skipped,
                key=lambda skip: skip_order_key(skip, manifest.config),
            )
        )
        != manifest.skipped
    ):
        raise DatasetValidationError("mutant skips are not in stable order")

    configured = tuple(manifest.config.operator_families)
    if not configured or len(set(configured)) != len(configured):
        raise DatasetValidationError(
            "generation config operator families are invalid"
        )
    _validate_coordinate_partition(
        records=records,
        skipped=manifest.skipped,
        config=manifest.config,
        canonical_by_id=canonical_by_id,
    )
    identities: set[str] = set()
    programs: set[tuple[str, str]] = set()
    coordinates: set[tuple[str, OperatorFamily, int]] = set()
    canonical_outcomes: dict[str, tuple[ExecutionOutcome, ...]] = {}
    canonical_content: dict[
        str,
        tuple[str, str, str, str, tuple[str, ...]],
    ] = {}
    for record in records:
        canonical = (
            None
            if canonical_by_id is None
            else canonical_by_id.get(record.task_id)
        )
        expected_identity = identity_hash_for(
            schema=_RECORD_SCHEMA,
            payload=record.model_dump(
                mode="json",
                exclude={"content_identity"},
            ),
        )
        if record.content_identity != expected_identity:
            raise DatasetValidationError(
                f"record content identity mismatch: {record.task_id}"
            )
        if record.content_identity in identities:
            raise DatasetValidationError("duplicate record content identity")
        identities.add(record.content_identity)
        _validate_record(
            record,
            manifest.config,
            canonical,
        )
        prior_outcomes = canonical_outcomes.setdefault(
            record.task_id,
            record.canonical_expected,
        )
        if prior_outcomes != record.canonical_expected:
            raise DatasetValidationError(
                "records disagree on canonical outcomes"
            )
        record_canonical_content = (
            record.prompt,
            record.entry_point,
            record.canonical_full_source,
            record.canonical_test,
            record.input_reprs,
        )
        prior_content = canonical_content.setdefault(
            record.task_id,
            record_canonical_content,
        )
        if prior_content != record_canonical_content:
            raise DatasetValidationError(
                "records disagree on canonical task content"
            )
        program_key = (record.task_id, record.mutated_full_source)
        if program_key in programs:
            raise DatasetValidationError("duplicate mutated program")
        programs.add(program_key)
        coordinate = (
            record.task_id,
            record.operator_family,
            record.seed,
        )
        if coordinate in coordinates:
            raise DatasetValidationError(
                "duplicate accepted search coordinate"
            )
        coordinates.add(coordinate)

    skip_coordinates: set[tuple[str, object, int | None]] = set()
    for skip in manifest.skipped:
        canonical = (
            None
            if canonical_by_id is None
            else canonical_by_id.get(skip.task_id)
        )
        coordinate = (
            skip.task_id,
            skip.operator_family,
            skip.seed,
        )
        if coordinate in skip_coordinates:
            raise DatasetValidationError("duplicate skipped search coordinate")
        skip_coordinates.add(coordinate)
        _validate_skip(
            skip,
            manifest.config,
            canonical,
        )

    actual_counts = tuple(
        FamilyCount(
            operator_family=family,
            count=sum(record.operator_family is family for record in records),
        )
        for family in sorted(configured, key=lambda item: item.value)
    )
    if manifest.accepted_by_family != actual_counts:
        raise DatasetValidationError("manifest family counts mismatch")


def _validate_record(
    record: MutantRecord,
    config: GenerationConfig,
    canonical: CanonicalTask | None,
) -> None:
    if record.operator_family not in config.operator_families:
        raise DatasetValidationError(
            "record operator family is absent from generation config"
        )
    if not 0 <= record.seed < config.seeds:
        raise DatasetValidationError("record seed is outside config")
    if record.site_node_path < 0 or record.site_target_index < 0:
        raise DatasetValidationError("record site address is negative")
    if record.task_id not in config.task_ids:
        raise DatasetValidationError("record task is outside config")
    count = len(record.input_reprs)
    if count > config.max_inputs_per_mutant:
        raise DatasetValidationError("record input count exceeds config")
    if canonical is not None and (
        record.prompt != canonical.prompt
        or record.entry_point != canonical.entry_point
        or record.canonical_full_source != canonical.canonical_full_source
        or record.canonical_test != canonical.canonical_test
        or record.input_reprs != canonical.input_reprs
    ):
        raise DatasetValidationError(
            "record canonical task/input content mismatch"
        )
    if (
        len(record.mutant_expected) != count
        or len(record.canonical_expected) != count
    ):
        raise DatasetValidationError("record outcome count mismatch")
    actual_distinct = tuple(
        index
        for index, (canonical, mutant) in enumerate(
            zip(
                record.canonical_expected,
                record.mutant_expected,
                strict=True,
            )
        )
        if canonical != mutant
    )
    if not actual_distinct:
        raise DatasetValidationError("record has no canonical divergence")
    if record.distinct_input_indices != actual_distinct:
        raise DatasetValidationError(
            "record distinct input indices are invalid"
        )
    for input_repr in record.input_reprs:
        try:
            input_value = ast.literal_eval(input_repr)
        except (SyntaxError, ValueError) as exc:
            raise DatasetValidationError(
                "record input is not a Python literal"
            ) from exc
        if not isinstance(input_value, tuple):
            raise DatasetValidationError(
                "record input is not an argument tuple"
            )
    try:
        sites = iter_sites(
            record.canonical_full_source,
            record.operator_family,
        )
    except MutationError as exc:
        raise DatasetValidationError(
            "record canonical source is malformed"
        ) from exc
    matches = tuple(
        site
        for site in sites
        if site.node_path == record.site_node_path
        and site.target_index == record.site_target_index
    )
    if len(matches) != 1:
        raise DatasetValidationError("record mutation site is not applicable")
    site = matches[0]
    if site.description != record.site_description:
        raise DatasetValidationError(
            "record mutation site description mismatch"
        )
    try:
        expected_mutant = apply_site(record.canonical_full_source, site)
    except MutationError as exc:
        raise DatasetValidationError(
            "record mutation could not be reproduced"
        ) from exc
    if expected_mutant != record.mutated_full_source:
        raise DatasetValidationError(
            "record mutant does not match its mutation site"
        )
    if not _defines_entry_point(
        record.canonical_full_source,
        record.entry_point,
    ):
        raise DatasetValidationError(
            "record canonical source does not define its entry point"
        )


def _validate_config(
    config: GenerationConfig,
    *,
    require_current_runtime_identity: bool,
) -> None:
    if config.dataset_id != HF_DATASET_ID:
        raise DatasetValidationError("generation config dataset id mismatch")
    if config.dataset_revision != HF_REVISION:
        raise DatasetValidationError(
            "generation config dataset revision mismatch"
        )
    if config.seeds < 1:
        raise DatasetValidationError("generation config seeds are invalid")
    if config.max_inputs_per_mutant < 1:
        raise DatasetValidationError(
            "generation config input limit is invalid"
        )
    if config.timeout_seconds <= 0 or not math.isfinite(
        config.timeout_seconds
    ):
        raise DatasetValidationError("generation config timeout is invalid")
    if len(set(config.task_ids)) != len(config.task_ids):
        raise DatasetValidationError(
            "generation config task ids contain duplicates"
        )
    if not config.task_ids:
        raise DatasetValidationError("generation config has no task ids")
    expected_families = tuple(
        family for family in ALL_FAMILIES if family in config.operator_families
    )
    if config.operator_families != expected_families:
        raise DatasetValidationError(
            "generation config operator family order is invalid"
        )
    if tuple(sorted(config.task_ids, key=_task_order_key)) != config.task_ids:
        raise DatasetValidationError(
            "generation config task id order is invalid"
        )
    if config.runner_identity == EXECUTOR_IDENTITY:
        if not _is_sha256(config.runtime_identity):
            raise DatasetValidationError(
                "production runner runtime identity is invalid"
            )
        if (
            require_current_runtime_identity
            and config.runtime_identity != current_runtime_identity()
        ):
            raise DatasetValidationError(
                "production runner runtime identity mismatch"
            )
    elif not config.runner_identity or not config.runtime_identity:
        raise DatasetValidationError(
            "injected runner provenance is incomplete"
        )


def _validate_skip(
    skip: SkippedMutation,
    config: GenerationConfig,
    canonical: CanonicalTask | None,
) -> None:
    if skip.task_id not in config.task_ids:
        raise DatasetValidationError("skipped task is outside config")
    if skip.operator_family == "*":
        if skip.seed is not None:
            raise DatasetValidationError("task-wide skip must not have a seed")
        if (
            canonical is not None
            and canonical.preparation_failure != skip.reason
        ):
            raise DatasetValidationError(
                "task-wide skip does not match preparation failure"
            )
        return
    if skip.operator_family not in config.operator_families:
        raise DatasetValidationError(
            "skipped operator family is outside config"
        )
    if skip.seed is None or not 0 <= skip.seed < config.seeds:
        raise DatasetValidationError("skipped seed is outside config")


def _validate_coordinate_partition(
    *,
    records: tuple[MutantRecord, ...],
    skipped: tuple[SkippedMutation, ...],
    config: GenerationConfig,
    canonical_by_id: dict[str, CanonicalTask] | None,
) -> None:
    accepted = {
        (record.task_id, record.operator_family, record.seed)
        for record in records
    }
    ordinary_skips = {
        (skip.task_id, skip.operator_family, skip.seed)
        for skip in skipped
        if skip.operator_family != "*"
    }
    wildcards = {
        skip.task_id for skip in skipped if skip.operator_family == "*"
    }
    if accepted & ordinary_skips:
        raise DatasetValidationError(
            "accepted and skipped coordinates overlap"
        )
    concrete_counts = dict.fromkeys(config.task_ids, 0)
    for concrete_coordinates in (accepted, ordinary_skips):
        for task_id, _family, _seed in concrete_coordinates:
            if task_id in concrete_counts:
                concrete_counts[task_id] += 1
    expected_concrete_count = len(config.operator_families) * config.seeds
    for task_id in config.task_ids:
        concrete_count = concrete_counts[task_id]
        has_preparation_failure = (
            task_id in wildcards
            if canonical_by_id is None
            else canonical_by_id[task_id].preparation_failure is not None
        )
        if has_preparation_failure:
            if task_id not in wildcards:
                raise DatasetValidationError(
                    "failed task requires exactly one task-wide skip"
                )
            if concrete_count:
                raise DatasetValidationError(
                    "task-wide skip overlaps concrete coordinates"
                )
        elif task_id in wildcards:
            raise DatasetValidationError(
                "prepared task must not have a task-wide skip"
            )
        elif concrete_count != expected_concrete_count:
            raise DatasetValidationError(
                f"incomplete coordinate partition for {task_id}"
            )


def _defines_entry_point(source: str, entry_point: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    )


def _task_order_key(task_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = task_id.rpartition("/")
    try:
        task_number = int(suffix)
    except ValueError:
        task_number = -1
    return (
        prefix if separator else task_id,
        task_number,
        task_id,
    )


def _encode_records(records: tuple[MutantRecord, ...]) -> bytes:
    lines = (
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    )
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def _decode_records(content: bytes) -> tuple[MutantRecord, ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("mutants.jsonl is not UTF-8") from exc
    if text and not text.endswith("\n"):
        raise DatasetValidationError("mutants.jsonl must end with a newline")
    records: list[MutantRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise DatasetValidationError(
                f"mutants.jsonl line {line_number} is blank"
            )
        try:
            records.append(MutantRecord.model_validate_json(line))
        except ValidationError as exc:
            raise DatasetValidationError(
                f"invalid mutants.jsonl line {line_number}"
            ) from exc
    return tuple(records)


def _read_authenticated_records(
    path: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> bytes:
    try:
        with stable_file(
            path,
            label=RECORDS_FILENAME,
            max_bytes=max_bytes,
        ) as captured:
            _validate_records_sha256(
                actual_sha256=captured.sha256,
                expected_sha256=expected_sha256,
            )
            return captured.path.read_bytes()
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc


def _read_bounded_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> bytes:
    try:
        with stable_file(path, label=label, max_bytes=max_bytes) as captured:
            return captured.path.read_bytes()
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc


def _validate_byte_ceiling(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_records_sha256(
    *,
    actual_sha256: str,
    expected_sha256: str,
) -> None:
    if actual_sha256 != expected_sha256:
        raise DatasetValidationError("mutants.jsonl SHA-256 mismatch")


def _encode_manifest(manifest: DatasetManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _dataset_identity(
    *,
    config_identity: str,
    records_sha256: str,
    accepted_count: int,
    accepted_by_family: tuple[FamilyCount, ...],
    skipped: tuple[SkippedMutation, ...],
) -> str:
    return identity_hash_for(
        schema=_DATASET_SCHEMA,
        payload={
            "accepted_count": accepted_count,
            "accepted_by_family": [
                item.model_dump(mode="json") for item in accepted_by_family
            ],
            "config_identity": config_identity,
            "records_sha256": records_sha256,
            "skipped": [item.model_dump(mode="json") for item in skipped],
        },
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = (
    "DATASET_SCHEMA_VERSION",
    "GENERATOR_VERSION",
    "DatasetArtifacts",
    "DatasetManifest",
    "DatasetValidationError",
    "GeneratedDataset",
    "GenerationConfig",
    "LoadedDataset",
    "MutantRecord",
    "build_record",
    "load_dataset",
    "publish_dataset",
)
