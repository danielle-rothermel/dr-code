"""Strict experiment-owned records and canonical atomic JSONL storage."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, BinaryIO, Final, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from dr_code.classifier.aggregation import (
    RepeatFailure,
    RepeatFailureKind,
    RepeatOutcome,
    RepeatPhase,
    aggregate_repeats,
)
from dr_code.classifier.taxonomy import FailureFamily, is_valid_label
from dr_code.eval.identity import identity_hash_for

DETAIL_ARTIFACT_VERSION: Final = "failure-classifications-v4"
DETAIL_SCHEMA_VERSION: Final = 4
EXPERIMENT_IDENTITY_SCHEMA: Final = "dr_code.failure_classifier.experiment"

Nonblank = Annotated[
    str, StringConstraints(strip_whitespace=False, min_length=1)
]
Rationale = Annotated[
    str,
    StringConstraints(strip_whitespace=False, min_length=1, max_length=280),
]
AuditDetail = Annotated[
    str,
    StringConstraints(strip_whitespace=False, min_length=1, max_length=512),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _StrictRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def validate_trimmed_strings(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() != value:
            raise ValueError("persisted strings must be trimmed")
        return value


class RunScopeRecord(_StrictRecord):
    run_id: Nonblank
    dataset_id: Nonblank
    corpus_sha256: Sha256
    preprocessing_manifest_sha256: Sha256
    preprocessing_identity: Sha256
    preprocessing_schema_version: int = Field(ge=1)
    definition_id: Nonblank
    definition_version: Nonblank
    definition_identity: Sha256
    evaluation_manifest_sha256: Sha256 | None
    evaluation_generation_id: Nonblank | None
    evaluation_pointer_sha256: Sha256 | None
    evaluation_identity: Sha256 | None

    @model_validator(mode="after")
    def validate_evaluation_coordinates(self) -> RunScopeRecord:
        coordinates = (
            self.evaluation_manifest_sha256,
            self.evaluation_generation_id,
            self.evaluation_pointer_sha256,
            self.evaluation_identity,
        )
        if any(value is None for value in coordinates) and any(
            value is not None for value in coordinates
        ):
            raise ValueError(
                "evaluation coordinates must be either all present or all null"
            )
        return self


class ClassifierConfigRecord(_StrictRecord):
    artifact_version: Literal["failure-classifications-v4"]
    schema_version: Literal[4]
    extraction_version: Nonblank
    aggregation_version: Nonblank
    taxonomy_version: Nonblank
    taxonomy_identity: Sha256
    prompt_version: Nonblank
    prompt_template_version: Nonblank
    prompt_template_identity: Sha256
    prompt_max_evidence_chars: int = Field(ge=1)
    prompt_max_input_chars: int = Field(ge=1)
    prompt_max_task_context_chars: int = Field(ge=1)
    prompt_max_metadata_chars: int = Field(ge=1)
    prompt_correction_attempts: int = Field(ge=0)
    provider: Nonblank
    model: Nonblank
    lane_policy_identity: Sha256
    lane_adapter: Nonblank
    lane_executable: Nonblank | None
    lane_timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )
    repeats: int = Field(ge=1)


class SelectionPolicyRecord(_StrictRecord):
    parse_limit: int | None = Field(default=None, ge=1)
    test_limit: int | None = Field(default=None, ge=1)


class ClassifierExperimentRecord(_StrictRecord):
    run: RunScopeRecord
    config: ClassifierConfigRecord
    selection: SelectionPolicyRecord


class ItemIdentityRecord(_StrictRecord):
    family: FailureFamily
    sample_id: Nonblank
    candidate_id: Nonblank | None
    evaluation_key: Sha256 | None
    task_id: Nonblank | None
    task_identity: Sha256 | None
    rendered_input_sha256: Sha256

    @model_validator(mode="after")
    def validate_family_coordinates(self) -> ItemIdentityRecord:
        if self.family is FailureFamily.PARSE and (
            self.candidate_id is not None or self.evaluation_key is not None
        ):
            raise ValueError(
                "parse identity cannot contain candidate coordinates"
            )
        if self.family is FailureFamily.TEST and (
            self.candidate_id is None
            or self.evaluation_key is None
            or self.task_id is None
            or self.task_identity is None
        ):
            raise ValueError(
                "test identity requires candidate, evaluation, and task "
                "coordinates"
            )
        if (self.task_id is None) != (self.task_identity is None):
            raise ValueError(
                "task_id and task_identity must be either both present or "
                "both null"
            )
        return self


class ResumeIdentityRecord(_StrictRecord):
    experiment_identity: Sha256
    repeats: int = Field(ge=1)
    item: ItemIdentityRecord


class RepeatFailureRecord(_StrictRecord):
    kind: RepeatFailureKind
    detail: AuditDetail


class RepeatRecord(_StrictRecord):
    index: int = Field(ge=0)
    label: Nonblank | None
    rationale: Rationale | None
    failure: RepeatFailureRecord | None
    phase: RepeatPhase
    attempt: Literal[1, 2]
    corrected: bool
    primary_validation_failure: AuditDetail | None

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str | None) -> str | None:
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError("rationale must be one line")
        return value

    @model_validator(mode="after")
    def validate_result_or_failure(self) -> RepeatRecord:
        succeeded = self.label is not None and self.rationale is not None
        if succeeded == (self.failure is not None):
            raise ValueError(
                "repeat record must contain either a response or a failure"
            )
        if not succeeded and (
            self.label is not None or self.rationale is not None
        ):
            raise ValueError(
                "failed repeat record cannot contain partial response fields"
            )
        if self.phase is RepeatPhase.PRIMARY:
            if (
                self.attempt != 1
                or self.corrected
                or self.primary_validation_failure is not None
            ):
                raise ValueError(
                    "primary repeat record has invalid correction metadata"
                )
        elif (
            self.attempt != 2
            or self.primary_validation_failure is None
            or self.corrected != succeeded
        ):
            raise ValueError(
                "correction repeat record has invalid correction metadata"
            )
        return self


class AggregateRecord(_StrictRecord):
    label: Nonblank | None
    agreement: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    tie: bool
    successful_repeats: int = Field(ge=0)
    failed_repeats: int = Field(ge=0)
    label_counts: dict[Nonblank, Annotated[int, Field(gt=0)]]

    @model_validator(mode="after")
    def validate_counts(self) -> AggregateRecord:
        if sum(self.label_counts.values()) != self.successful_repeats:
            raise ValueError("label counts must total successful repeats")
        if self.successful_repeats == 0:
            if (
                self.label is not None
                or self.agreement is not None
                or self.tie
            ):
                raise ValueError(
                    "all-failed aggregate cannot contain a machine verdict"
                )
        elif self.label is None or self.agreement is None:
            raise ValueError(
                "successful aggregate requires label and agreement"
            )
        return self


class ItemRecord(_StrictRecord):
    record_type: Literal["item"] = "item"
    identity: ResumeIdentityRecord
    aggregate: AggregateRecord
    repeats: tuple[RepeatRecord, ...]

    @model_validator(mode="after")
    def validate_derived_aggregate(self) -> ItemRecord:
        expected_repeats = self.identity.repeats
        if len(self.repeats) != expected_repeats:
            raise ValueError("repeat records must match configured repeats")
        if tuple(record.index for record in self.repeats) != tuple(
            range(expected_repeats)
        ):
            raise ValueError("repeat record indices must be contiguous")
        outcomes = tuple(
            RepeatOutcome(
                label=record.label,
                rationale=record.rationale,
                failure=(
                    RepeatFailure(
                        kind=record.failure.kind,
                        detail=record.failure.detail,
                    )
                    if record.failure is not None
                    else None
                ),
                phase=record.phase,
                attempt=record.attempt,
                corrected=record.corrected,
                primary_validation_failure=record.primary_validation_failure,
            )
            for record in self.repeats
        )
        derived = aggregate_repeats(outcomes)
        family = self.identity.item.family
        if any(
            record.label is not None
            and not is_valid_label(family, record.label)
            for record in self.repeats
        ):
            raise ValueError("repeat label is outside the item taxonomy")
        if self.aggregate.label is not None and not is_valid_label(
            family, self.aggregate.label
        ):
            raise ValueError("aggregate label is outside the item taxonomy")
        if (
            self.aggregate.label != derived.label
            or self.aggregate.agreement != derived.agreement
            or self.aggregate.tie != derived.tie
            or self.aggregate.successful_repeats != derived.successful_repeats
            or self.aggregate.failed_repeats != derived.failed_repeats
            or self.aggregate.label_counts != derived.label_counts
        ):
            raise ValueError("aggregate does not match repeat records")
        return self


class ExperimentHeaderRecord(_StrictRecord):
    record_type: Literal["experiment"] = "experiment"
    artifact_version: Literal["failure-classifications-v4"]
    schema_version: Literal[4]
    experiment_identity: Sha256
    experiment: ClassifierExperimentRecord

    @model_validator(mode="after")
    def validate_identity(self) -> ExperimentHeaderRecord:
        if self.experiment_identity != experiment_identity(self.experiment):
            raise ValueError("classification experiment identity mismatch")
        return self


def canonical_record_bytes(record: BaseModel) -> bytes:
    value = record.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def identity_key(identity: ResumeIdentityRecord) -> bytes:
    return json.dumps(
        identity.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def experiment_identity(experiment: ClassifierExperimentRecord) -> str:
    """Hash one typed, complete, recursively finite experiment payload."""
    return identity_hash_for(
        schema=EXPERIMENT_IDENTITY_SCHEMA,
        payload=experiment.model_dump(mode="json"),
    )


def read_artifact_stream(
    lines: Iterable[bytes],
    *,
    on_record: Callable[[ItemRecord], None],
) -> ExperimentHeaderRecord:
    """Validate one JSONL stream and deliver records incrementally."""
    raw_lines = iter(lines)
    try:
        first = next(raw_lines)
    except StopIteration:
        raise ValueError("classification details are empty")
    first_line = _decode_artifact_line(first)
    header = cast(
        ExperimentHeaderRecord,
        _load_line(
            first_line,
            line_number=1,
            record_type=ExperimentHeaderRecord,
        ),
    )
    for line_number, raw_line in enumerate(raw_lines, start=2):
        line = _decode_artifact_line(raw_line)
        record = cast(
            ItemRecord,
            _load_line(
                line,
                line_number=line_number,
                record_type=ItemRecord,
            ),
        )
        if record.identity.experiment_identity != header.experiment_identity:
            raise ValueError(
                f"classification details line {line_number} belongs to "
                "a different experiment"
            )
        if record.identity.repeats != header.experiment.config.repeats:
            raise ValueError(
                f"classification details line {line_number} has a "
                "different repeat policy"
            )
        on_record(record)
    return header


def _decode_artifact_line(raw_line: bytes) -> str:
    try:
        line = raw_line.rstrip(b"\r\n").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("classification details are not UTF-8") from exc
    if not line:
        raise ValueError("classification details contain a blank JSONL record")
    return line


def write_records_atomic(
    path: Path,
    experiment: ClassifierExperimentRecord,
    records: Iterable[ItemRecord],
    *,
    records_are_sorted: bool = False,
) -> None:
    experiment_sha256 = experiment_identity(experiment)
    header = ExperimentHeaderRecord(
        artifact_version=DETAIL_ARTIFACT_VERSION,
        schema_version=DETAIL_SCHEMA_VERSION,
        experiment_identity=experiment_sha256,
        experiment=experiment,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if records_are_sorted:
        _write_sorted_records_atomic(
            path,
            header,
            records,
            experiment_sha256=experiment_sha256,
            repeats=experiment.config.repeats,
        )
        return
    sorter_descriptor, sorter_raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.records.",
        suffix=".sqlite3",
        dir=path.parent,
    )
    os.close(sorter_descriptor)
    sorter_path = Path(sorter_raw_path)
    try:
        with sqlite3.connect(sorter_path) as sorter:
            sorter.execute(
                "CREATE TABLE records "
                "(identity BLOB PRIMARY KEY, payload BLOB NOT NULL)"
            )
            for record in records:
                if record.identity.experiment_identity != experiment_sha256:
                    raise ValueError(
                        "classification record belongs to a different "
                        "experiment"
                    )
                if record.identity.repeats != experiment.config.repeats:
                    raise ValueError(
                        "classification record has a different repeat policy"
                    )
                try:
                    sorter.execute(
                        "INSERT INTO records(identity, payload) VALUES (?, ?)",
                        (
                            identity_key(record.identity),
                            canonical_record_bytes(record),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "duplicate classification record identity"
                    ) from exc
            with _atomic_temp_stream(path) as stream:
                stream.write(canonical_record_bytes(header) + b"\n")
                cursor = sorter.execute(
                    "SELECT payload FROM records ORDER BY identity"
                )
                while rows := cursor.fetchmany(128):
                    for (payload,) in rows:
                        stream.write(cast(bytes, payload) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        try:
            sorter_path.unlink()
        except FileNotFoundError:
            pass


def _write_sorted_records_atomic(
    path: Path,
    header: ExperimentHeaderRecord,
    records: Iterable[ItemRecord],
    *,
    experiment_sha256: str,
    repeats: int,
) -> None:
    with _atomic_temp_stream(path) as stream:
        stream.write(canonical_record_bytes(header) + b"\n")
        previous_key: bytes | None = None
        for record in records:
            if record.identity.experiment_identity != experiment_sha256:
                raise ValueError(
                    "classification record belongs to a different experiment"
                )
            if record.identity.repeats != repeats:
                raise ValueError(
                    "classification record has a different repeat policy"
                )
            key = identity_key(record.identity)
            if previous_key is not None and key <= previous_key:
                raise ValueError(
                    "classification records are not uniquely sorted"
                )
            previous_key = key
            stream.write(canonical_record_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _atomic_temp_stream(path: Path) -> Iterator[BinaryIO]:
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(raw_temp_path)
    try:
        try:
            stream = os.fdopen(descriptor, "wb", closefd=True)
        except BaseException:
            os.close(descriptor)
            raise
        with stream:
            yield stream
        os.replace(temp_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_line(
    line: str,
    *,
    line_number: int,
    record_type: type[ExperimentHeaderRecord] | type[ItemRecord],
) -> ExperimentHeaderRecord | ItemRecord:
    try:
        payload = json.loads(
            line,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        return record_type.model_validate_json(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            strict=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"invalid classification details line {line_number}: {exc}"
        ) from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"invalid JSON number: {value}")


__all__ = (
    "DETAIL_ARTIFACT_VERSION",
    "DETAIL_SCHEMA_VERSION",
    "AggregateRecord",
    "ClassifierExperimentRecord",
    "ClassifierConfigRecord",
    "ExperimentHeaderRecord",
    "ItemIdentityRecord",
    "ItemRecord",
    "RepeatFailureRecord",
    "RepeatRecord",
    "ResumeIdentityRecord",
    "RunScopeRecord",
    "SelectionPolicyRecord",
    "canonical_record_bytes",
    "experiment_identity",
    "identity_key",
    "read_artifact_stream",
    "write_records_atomic",
)
