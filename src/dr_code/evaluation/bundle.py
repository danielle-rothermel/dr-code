from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from dr_serialize import (
    Jsonable,
    Sha256Digest,
    canonical_json_bytes,
    decode_strict_json_bytes,
)
from dr_store import (
    ArtifactBundlePublication,
    ArtifactBundleReader,
    BundleReadLimits,
    ObjectStore,
    VerifyingArtifactReader,
)
from pydantic import Field, PositiveInt, TypeAdapter

from dr_code.core.models import FrozenModel
from dr_code.evaluation.batch import (
    EvaluationBatchRequest,
    EvaluationProjectionReference,
    ProjectionKind,
)
from dr_code.evaluation.identity import EvaluationAttemptIdentity
from dr_code.evaluation.projections import ProjectionRow
from dr_code.evaluation.records import (
    EvaluationAttemptRecord,
    EvaluatedSampleRecord,
    SAMPLE_EVALUATION_RECORD_ADAPTER,
    SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION,
    ReusedCandidateProvenance,
    SampleEvaluationRecord,
)
from dr_code.evaluation.references import (
    BundleRecordReference,
    EvidenceReference,
    StoredRecordReference,
)
from dr_code.evaluation.validation import (
    validate_evaluation_attempt_graph,
    validate_sample_record_graph,
)

EVALUATION_BUNDLE_SCHEMA_VERSION: Final = 1
EVALUATION_PROJECTION_SCHEMA_VERSION: Final = 1
EVALUATION_BUNDLE_FORMAT: Final = "dr-code-evaluation-bundle-v1"
EVALUATION_PROJECTION_FORMAT: Final = "dr-code-evaluation-projection-v1"
SAMPLE_RECORD_OBJECT_SCHEMA: Final = "dr-code/sample-evaluation-record-v1"
EVALUATION_ATTEMPT_ARTIFACT: Final = "evaluation-attempt.json"
SAMPLE_RECORD_SHARD_FORMAT: Final = "dr-code-sample-record-shard-v1"
SAMPLE_REFERENCE_SHARD_FORMAT: Final = "dr-code-sample-reference-shard-v1"

# Persisted names are explicit wire contracts. Do not construct this mapping by
# iterating ProjectionKind.
_PROJECTION_ARTIFACT_NAMES: Final = {
    ProjectionKind.EVALUATION_SAMPLES: "projection-evaluation-samples.jsonl",
    ProjectionKind.MATERIALIZED_CANDIDATES: (
        "projection-materialized-candidates.jsonl"
    ),
    ProjectionKind.METRIC_RECORDS: "projection-metric-records.jsonl",
    ProjectionKind.AGGREGATION_RESULTS: (
        "projection-aggregation-results.jsonl"
    ),
    ProjectionKind.SCORES: "projection-scores.jsonl",
}


class EvaluationBundlePayload(FrozenModel):
    format: Literal["dr-code-evaluation-bundle-v1"] = EVALUATION_BUNDLE_FORMAT
    schema_version: Literal[1] = EVALUATION_BUNDLE_SCHEMA_VERSION
    attempt: EvaluationAttemptIdentity
    attempt_artifact: Literal["evaluation-attempt.json"] = (
        EVALUATION_ATTEMPT_ARTIFACT
    )
    projections: tuple[EvaluationProjectionReference, ...]


class ProjectionArtifactHeader(FrozenModel):
    format: Literal["dr-code-evaluation-projection-v1"] = (
        EVALUATION_PROJECTION_FORMAT
    )
    schema_version: Literal[1] = EVALUATION_PROJECTION_SCHEMA_VERSION
    source_attempt: EvaluationAttemptIdentity
    kind: ProjectionKind
    definition_version: Literal[1] = 1


class EvaluationReadLimits(FrozenModel):
    bundle: BundleReadLimits
    max_sample_records: PositiveInt
    max_object_reads: PositiveInt
    max_reference_depth: PositiveInt


class RestoredEvaluationAttempt(FrozenModel):
    attempt: EvaluationAttemptRecord
    samples: tuple[SampleEvaluationRecord, ...]


class EvaluationBundleAudit(FrozenModel):
    attempt: EvaluationAttemptIdentity
    artifact_count: int = Field(ge=0)
    sample_record_count: int = Field(ge=0)
    object_read_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _PlacedRecord:
    reference: EvidenceReference
    record: SampleEvaluationRecord


@dataclass(frozen=True, slots=True)
class _EncodedRecord:
    payload: bytes
    sha256: str


class _RecordPlacement:
    def __init__(
        self,
        request: EvaluationBatchRequest,
        *,
        publication: ArtifactBundlePublication | None,
        object_store: ObjectStore | None,
    ) -> None:
        self._request = request
        self._publication = publication
        self._object_store = object_store
        self._shard_index = 0
        self._shard_records: list[_EncodedRecord] = []
        self._shard_references: list[StoredRecordReference] = []
        self._placed_references: list[EvidenceReference] = []
        self._finished = False

    async def place(
        self, record: SampleEvaluationRecord, /
    ) -> EvidenceReference:
        if self._finished:
            raise RuntimeError("record placement is already finished")
        expected = next(
            (
                item
                for item in self._request.inputs
                if item.slot == record.slot
            ),
            None,
        )
        if expected is None:
            raise ValueError(
                "sample record slot does not belong to the batch request"
            )
        validate_sample_record_graph(
            record,
            slot=expected.slot,
            sample=expected.sample.metadata.identity,
            plan=self._request.plan,
            runtime=self._request.runtime,
            cache_namespace=self._request.cache_namespace,
        )
        payload = canonical_json_bytes(record.model_dump(mode="json"))
        if self._request.record_placement.value == "bundle_local":
            reference = await self._place_bundle_record(payload)
        else:
            if self._object_store is None:
                raise ValueError(
                    "object-store placement requires object_store"
                )
            object_reference, _ = await self._object_store.put(
                SAMPLE_RECORD_OBJECT_SCHEMA,
                record.model_dump(mode="json"),
            )
            reference = StoredRecordReference(
                reference=object_reference,
                schema_version=SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION,
            )
            if self._publication is not None:
                await self._admit_reference(reference)
        self._placed_references.append(reference)
        return reference

    async def _place_bundle_record(
        self, payload: bytes
    ) -> BundleRecordReference:
        if self._publication is None:
            raise ValueError("bundle-local placement requires publication")
        encoded = _EncodedRecord(
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        if not self._can_admit_record(encoded):
            if not self._shard_records:
                raise ValueError(
                    "one sample record exceeds shard max_uncompressed_bytes"
                )
            await self._flush_record_shard()
            if not self._can_admit_record(encoded):
                raise ValueError(
                    "one sample record exceeds shard max_uncompressed_bytes"
                )
        artifact_name = _sample_record_artifact_name(self._shard_index)
        record_index = len(self._shard_records)
        self._shard_records.append(encoded)
        reference = BundleRecordReference(
            artifact_name=artifact_name,
            record_index=record_index,
            record_sha256=Sha256Digest(encoded.sha256),
            schema=SAMPLE_RECORD_OBJECT_SCHEMA,
            schema_version=SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION,
        )
        if len(self._shard_records) >= self._resident_record_limit:
            await self._flush_record_shard()
        return reference

    async def _admit_reference(self, reference: StoredRecordReference) -> None:
        if not self._can_admit_reference(reference):
            if not self._shard_references:
                raise ValueError(
                    "one stored-record reference exceeds shard "
                    "max_uncompressed_bytes"
                )
            await self._flush_reference_shard()
            if not self._can_admit_reference(reference):
                raise ValueError(
                    "one stored-record reference exceeds shard "
                    "max_uncompressed_bytes"
                )
        self._shard_references.append(reference)
        if len(self._shard_references) >= self._resident_record_limit:
            await self._flush_reference_shard()

    @property
    def _resident_record_limit(self) -> int:
        return min(
            self._request.shard_limits.max_records,
            self._request.window_limits.max_record_assemblies,
        )

    def _can_admit_record(self, record: _EncodedRecord) -> bool:
        candidate = (*self._shard_records, record)
        return (
            len(candidate) <= self._resident_record_limit
            and len(_record_shard_bytes(self._request.attempt, candidate))
            <= self._request.shard_limits.max_uncompressed_bytes
        )

    def _can_admit_reference(self, reference: StoredRecordReference) -> bool:
        candidate = (*self._shard_references, reference)
        return (
            len(candidate) <= self._resident_record_limit
            and len(_reference_shard_bytes(self._request.attempt, candidate))
            <= self._request.shard_limits.max_uncompressed_bytes
        )

    async def _flush_record_shard(self) -> None:
        if not self._shard_records:
            return
        assert self._publication is not None
        name = _sample_record_artifact_name(self._shard_index)
        payload = _record_shard_bytes(
            self._request.attempt, tuple(self._shard_records)
        )
        await asyncio.to_thread(
            _write_artifact, self._publication, name, payload
        )
        self._shard_records.clear()
        self._shard_index += 1

    async def _flush_reference_shard(self) -> None:
        if not self._shard_references:
            return
        assert self._publication is not None
        name = _sample_reference_artifact_name(self._shard_index)
        payload = _reference_shard_bytes(
            self._request.attempt, tuple(self._shard_references)
        )
        await asyncio.to_thread(
            _write_artifact, self._publication, name, payload
        )
        self._shard_references.clear()
        self._shard_index += 1

    async def finish(self) -> None:
        if self._finished:
            return
        if self._request.record_placement.value == "bundle_local":
            await self._flush_record_shard()
        else:
            await self._flush_reference_shard()
        self._finished = True

    @property
    def references(self) -> tuple[EvidenceReference, ...]:
        if not self._finished:
            raise RuntimeError("record placement is not finished")
        return tuple(self._placed_references)

    async def iter_records(
        self,
    ) -> AsyncIterator[tuple[EvidenceReference, SampleEvaluationRecord]]:
        """Re-read placed evidence sequentially without retaining the attempt."""

        local_cache_name: str | None = None
        local_cache: tuple[SampleEvaluationRecord, ...] = ()
        for reference in self.references:
            if isinstance(reference, BundleRecordReference):
                assert self._publication is not None
                if reference.artifact_name != local_cache_name:
                    local_cache = await asyncio.to_thread(
                        _load_authored_sample_shard,
                        self._publication.path / reference.artifact_name,
                        self._request.attempt,
                    )
                    local_cache_name = reference.artifact_name
                record = local_cache[reference.record_index]
            else:
                assert self._object_store is not None
                decoded = await self._object_store.get(reference.reference)
                record = SAMPLE_EVALUATION_RECORD_ADAPTER.validate_json(
                    canonical_json_bytes(decoded), strict=True
                )
            yield reference, record
        local_cache = ()

    async def validate_bundle_reference_closure(self) -> None:
        available = set(self.references)
        async for _, record in self.iter_records():
            for reference, _ in _nested_evidence_references(record):
                if (
                    isinstance(reference, BundleRecordReference)
                    and reference not in available
                ):
                    raise ValueError(
                        "nested bundle record reference is not closed by the bundle"
                    )


def _sample_record_artifact_name(index: int) -> str:
    return f"sample-records-{index:08d}.jsonl"


def _sample_reference_artifact_name(index: int) -> str:
    return f"sample-record-references-{index:08d}.jsonl"


def _shard_header(
    *, format: str, attempt: EvaluationAttemptIdentity, record_count: int
) -> Jsonable:
    return {
        "format": format,
        "schema_version": 1,
        "source_attempt": attempt.model_dump(mode="json"),
        "record_count": record_count,
    }


def _record_shard_bytes(
    attempt: EvaluationAttemptIdentity,
    records: Iterable[_EncodedRecord],
) -> bytes:
    rows = tuple(records)
    header = canonical_json_bytes(
        _shard_header(
            format=SAMPLE_RECORD_SHARD_FORMAT,
            attempt=attempt,
            record_count=len(rows),
        )
    )
    return b"\n".join((header, *(row.payload for row in rows))) + b"\n"


def _reference_shard_bytes(
    attempt: EvaluationAttemptIdentity,
    references: Iterable[StoredRecordReference],
) -> bytes:
    rows = tuple(references)
    header = canonical_json_bytes(
        _shard_header(
            format=SAMPLE_REFERENCE_SHARD_FORMAT,
            attempt=attempt,
            record_count=len(rows),
        )
    )
    return (
        b"\n".join(
            (
                header,
                *(
                    canonical_json_bytes(reference.model_dump(mode="json"))
                    for reference in rows
                ),
            )
        )
        + b"\n"
    )


def _write_artifact(
    publication: ArtifactBundlePublication, name: str, payload: bytes
) -> None:
    writer = publication.open_artifact(name)
    writer.write(payload)
    writer.finalize()


def _load_authored_sample_shard(
    path: Path, attempt: EvaluationAttemptIdentity
) -> tuple[SampleEvaluationRecord, ...]:
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("authored sample shard is incomplete")
    lines = data[:-1].split(b"\n")
    decoded = tuple(
        decode_strict_json_bytes(
            line,
            max_bytes=max(1, len(line)),
            max_depth=100,
        )
        for line in lines
    )
    count = _validate_shard_header(
        decoded[0],
        expected_format=SAMPLE_RECORD_SHARD_FORMAT,
        attempt=attempt,
    )
    if count != len(decoded) - 1:
        raise ValueError("authored sample shard record_count is incorrect")
    return tuple(
        SAMPLE_EVALUATION_RECORD_ADAPTER.validate_json(
            canonical_json_bytes(row), strict=True
        )
        for row in decoded[1:]
    )


def _projection_header_payload(
    header: ProjectionArtifactHeader,
) -> Jsonable:
    return {
        "format": EVALUATION_PROJECTION_FORMAT,
        "schema_version": EVALUATION_PROJECTION_SCHEMA_VERSION,
        "source_attempt": header.source_attempt.model_dump(mode="json"),
        "kind": header.kind.value,
        "definition_version": header.definition_version,
    }


def _projection_artifact_bytes(
    header: ProjectionArtifactHeader, rows: Iterable[ProjectionRow]
) -> bytes:
    return (
        b"\n".join(
            (
                canonical_json_bytes(_projection_header_payload(header)),
                *(
                    canonical_json_bytes(row.model_dump(mode="json"))
                    for row in rows
                ),
            )
        )
        + b"\n"
    )


async def _write_projection_artifact(
    publication: ArtifactBundlePublication,
    header: ProjectionArtifactHeader,
    rows: AsyncIterator[ProjectionRow],
    *,
    max_resident_rows: int,
) -> None:
    name = _PROJECTION_ARTIFACT_NAMES[header.kind]
    writer = await asyncio.to_thread(publication.open_artifact, name)
    await asyncio.to_thread(
        writer.write,
        canonical_json_bytes(_projection_header_payload(header)) + b"\n",
    )
    buffered: list[bytes] = []
    async for row in rows:
        buffered.append(
            canonical_json_bytes(row.model_dump(mode="json")) + b"\n"
        )
        if len(buffered) >= max_resident_rows:
            payload = b"".join(buffered)
            buffered.clear()
            await asyncio.to_thread(writer.write, payload)
    if buffered:
        await asyncio.to_thread(writer.write, b"".join(buffered))
    await asyncio.to_thread(writer.finalize)


def _publish_bundle(
    publication: ArtifactBundlePublication,
    *,
    attempt: EvaluationAttemptRecord,
    projections: tuple[EvaluationProjectionReference, ...],
) -> None:
    _write_artifact(
        publication,
        EVALUATION_ATTEMPT_ARTIFACT,
        canonical_json_bytes(attempt.model_dump(mode="json")),
    )
    payload = EvaluationBundlePayload(
        attempt=attempt.identity,
        projections=projections,
    )
    publication.publish(
        {
            "format": EVALUATION_BUNDLE_FORMAT,
            "schema_version": EVALUATION_BUNDLE_SCHEMA_VERSION,
            "attempt": payload.attempt.model_dump(mode="json"),
            "attempt_artifact": EVALUATION_ATTEMPT_ARTIFACT,
            "projections": [
                projection.model_dump(mode="json")
                for projection in payload.projections
            ],
        }
    )


def _consume_bytes(reader: ArtifactBundleReader, name: str) -> bytes:
    captured: list[bytes] = []

    def consume(stream: VerifyingArtifactReader) -> None:
        captured.append(stream.read())

    reader.consume_and_verify_artifact(name, consume)
    return captured[0]


def _decode_canonical_lines(
    data: bytes,
    *,
    limits: EvaluationReadLimits,
) -> tuple[Jsonable, ...]:
    if not data.endswith(b"\n"):
        raise ValueError("JSONL artifact must end with a newline")
    raw_lines = data[:-1].split(b"\n")
    if not raw_lines or any(not line for line in raw_lines):
        raise ValueError("JSONL artifact contains an empty line")
    decoded: list[Jsonable] = []
    for line in raw_lines:
        value = decode_strict_json_bytes(
            line,
            max_bytes=limits.bundle.max_bytes_per_artifact,
            max_depth=limits.bundle.manifest_max_depth,
        )
        if canonical_json_bytes(value) != line:
            raise ValueError("JSONL value is not canonical JSON")
        decoded.append(value)
    return tuple(decoded)


_PROJECTION_ROW_ADAPTER: Final = TypeAdapter[ProjectionRow](ProjectionRow)


def _read_projection_artifact(
    reader: ArtifactBundleReader,
    kind: ProjectionKind,
    *,
    limits: EvaluationReadLimits,
) -> tuple[ProjectionArtifactHeader, tuple[ProjectionRow, ...]]:
    lines = _decode_canonical_lines(
        _consume_bytes(reader, _PROJECTION_ARTIFACT_NAMES[kind]),
        limits=limits,
    )
    header = ProjectionArtifactHeader.model_validate_json(
        canonical_json_bytes(lines[0]), strict=True
    )
    if header.kind is not kind:
        raise ValueError("projection artifact kind does not match its name")
    if len(lines) - 1 > limits.max_sample_records:
        raise ValueError("projection row count exceeds max_sample_records")
    rows = tuple(
        _PROJECTION_ROW_ADAPTER.validate_json(
            canonical_json_bytes(line), strict=True
        )
        for line in lines[1:]
    )
    if any(row.kind is not kind for row in rows):
        raise ValueError("projection row kind does not match its header")
    if any(row.source_attempt != header.source_attempt for row in rows):
        raise ValueError("projection row source attempt does not match header")
    return header, rows


def read_evaluation_projection(
    bundle_path: str | Path,
    kind: ProjectionKind,
    /,
    *,
    limits: EvaluationReadLimits,
) -> tuple[ProjectionArtifactHeader, tuple[ProjectionRow, ...]]:
    """Verify and decode only one fixed self-bound projection artifact."""

    reader = ArtifactBundleReader(bundle_path, limits=limits.bundle)
    return _read_projection_artifact(reader, kind, limits=limits)


def _decode_attempt(
    reader: ArtifactBundleReader, limits: EvaluationReadLimits
) -> EvaluationAttemptRecord:
    payload = _consume_bytes(reader, EVALUATION_ATTEMPT_ARTIFACT)
    decoded = decode_strict_json_bytes(
        payload,
        max_bytes=limits.bundle.max_bytes_per_artifact,
        max_depth=limits.bundle.manifest_max_depth,
    )
    if canonical_json_bytes(decoded) != payload:
        raise ValueError("evaluation attempt artifact is not canonical JSON")
    return EvaluationAttemptRecord.model_validate_json(
        canonical_json_bytes(decoded), strict=True
    )


def _decode_sample_shard(
    reader: ArtifactBundleReader,
    name: str,
    *,
    attempt: EvaluationAttemptIdentity,
    limits: EvaluationReadLimits,
) -> tuple[SampleEvaluationRecord, ...]:
    lines = _decode_canonical_lines(
        _consume_bytes(reader, name), limits=limits
    )
    header = _validate_shard_header(
        lines[0],
        expected_format=SAMPLE_RECORD_SHARD_FORMAT,
        attempt=attempt,
    )
    if header != len(lines) - 1:
        raise ValueError("sample shard record_count is incorrect")
    return tuple(
        SAMPLE_EVALUATION_RECORD_ADAPTER.validate_json(
            canonical_json_bytes(line), strict=True
        )
        for line in lines[1:]
    )


def _decode_reference_shard(
    reader: ArtifactBundleReader,
    name: str,
    *,
    attempt: EvaluationAttemptIdentity,
    limits: EvaluationReadLimits,
) -> tuple[StoredRecordReference, ...]:
    lines = _decode_canonical_lines(
        _consume_bytes(reader, name), limits=limits
    )
    header = _validate_shard_header(
        lines[0],
        expected_format=SAMPLE_REFERENCE_SHARD_FORMAT,
        attempt=attempt,
    )
    if header != len(lines) - 1:
        raise ValueError("reference shard record_count is incorrect")
    return tuple(
        StoredRecordReference.model_validate_json(
            canonical_json_bytes(line), strict=True
        )
        for line in lines[1:]
    )


def _validate_shard_header(
    payload: Jsonable,
    *,
    expected_format: str,
    attempt: EvaluationAttemptIdentity,
) -> int:
    if not isinstance(payload, dict) or set(payload) != {
        "format",
        "schema_version",
        "source_attempt",
        "record_count",
    }:
        raise ValueError("sample shard header has unsupported fields")
    if payload["format"] != expected_format or payload["schema_version"] != 1:
        raise ValueError(
            "sample shard format or schema version is unsupported"
        )
    if (
        EvaluationAttemptIdentity.model_validate_json(
            canonical_json_bytes(payload["source_attempt"]), strict=True
        )
        != attempt
    ):
        raise ValueError("sample shard source attempt does not match attempt")
    count = payload["record_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("sample shard record_count must be non-negative")
    return count


async def restore_evaluation_attempt(
    bundle_path: str | Path,
    /,
    *,
    object_store: ObjectStore | None,
    limits: EvaluationReadLimits,
) -> RestoredEvaluationAttempt:
    """Restore authoritative samples without a preliminary whole-bundle audit."""

    reader = ArtifactBundleReader(bundle_path, limits=limits.bundle)
    attempt = await asyncio.to_thread(_decode_attempt, reader, limits)
    samples, _ = await _restore_evaluation_records(
        reader,
        attempt,
        object_store=object_store,
        limits=limits,
    )
    validate_evaluation_attempt_graph(attempt, samples)
    return RestoredEvaluationAttempt(attempt=attempt, samples=samples)


async def _restore_evaluation_records(
    reader: ArtifactBundleReader,
    attempt: EvaluationAttemptRecord,
    *,
    object_store: ObjectStore | None,
    limits: EvaluationReadLimits,
) -> tuple[tuple[SampleEvaluationRecord, ...], int]:
    present = tuple(
        member for member in attempt.members if member.record is not None
    )
    if len(present) > limits.max_sample_records:
        raise ValueError("sample record count exceeds max_sample_records")

    local_by_name: dict[str, dict[int, SampleEvaluationRecord]] = {}
    local_names = tuple(
        dict.fromkeys(
            member.record.artifact_name
            for member in present
            if isinstance(member.record, BundleRecordReference)
        )
    )
    for name in local_names:
        records = await asyncio.to_thread(
            _decode_sample_shard,
            reader,
            name,
            attempt=attempt.identity,
            limits=limits,
        )
        if len(records) > limits.max_sample_records:
            raise ValueError("sample shard count exceeds max_sample_records")
        local_by_name[name] = {
            index: record for index, record in enumerate(records)
        }

    stored_references = tuple(
        member.record
        for member in present
        if isinstance(member.record, StoredRecordReference)
    )
    if stored_references:
        if object_store is None:
            raise ValueError("stored record restoration requires object_store")
        indexed: list[StoredRecordReference] = []
        shard = 0
        while len(indexed) < len(stored_references):
            indexed.extend(
                await asyncio.to_thread(
                    _decode_reference_shard,
                    reader,
                    _sample_reference_artifact_name(shard),
                    attempt=attempt.identity,
                    limits=limits,
                )
            )
            if len(indexed) > limits.max_sample_records:
                raise ValueError("reference index exceeds max_sample_records")
            shard += 1
        if tuple(indexed) != stored_references:
            raise ValueError(
                "stored-record reference index does not match attempt"
            )

    samples: list[SampleEvaluationRecord] = []
    object_reads = 0
    for member in present:
        reference = member.record
        assert reference is not None
        if isinstance(reference, BundleRecordReference):
            try:
                sample = local_by_name[reference.artifact_name][
                    reference.record_index
                ]
            except KeyError as error:
                raise ValueError(
                    "bundle record reference is out of range"
                ) from error
            encoded = canonical_json_bytes(sample.model_dump(mode="json"))
            if hashlib.sha256(encoded).hexdigest() != str(
                reference.record_sha256
            ):
                raise ValueError(
                    "bundle record reference hash does not match record"
                )
            if (
                reference.schema != SAMPLE_RECORD_OBJECT_SCHEMA
                or reference.schema_version
                != SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION
            ):
                raise ValueError(
                    "bundle record reference schema is unsupported"
                )
        else:
            object_reads += 1
            if object_reads > limits.max_object_reads:
                raise ValueError("stored record reads exceed max_object_reads")
            assert object_store is not None
            decoded = await object_store.get(reference.reference)
            sample = SAMPLE_EVALUATION_RECORD_ADAPTER.validate_json(
                canonical_json_bytes(decoded), strict=True
            )
            if (
                reference.reference.schema != SAMPLE_RECORD_OBJECT_SCHEMA
                or reference.schema_version
                != SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION
            ):
                raise ValueError(
                    "stored record reference schema is unsupported"
                )
        if (
            sample.slot != member.slot
            or sample.sample.identity != member.sample
        ):
            raise ValueError(
                "sample record identity does not match attempt member"
            )
        samples.append(sample)
    resolved = {
        member.record: sample
        for member, sample in zip(present, samples, strict=True)
        if member.record is not None
    }
    nested_reads = await _validate_nested_evidence_references(
        tuple(samples),
        resolved=resolved,
        object_store=object_store,
        limits=limits,
        initial_reference_count=len(present),
    )
    return tuple(samples), object_reads + nested_reads


async def _validate_nested_evidence_references(
    samples: tuple[SampleEvaluationRecord, ...],
    *,
    resolved: dict[EvidenceReference, SampleEvaluationRecord],
    object_store: ObjectStore | None,
    limits: EvaluationReadLimits,
    initial_reference_count: int,
) -> int:
    reference_count = initial_reference_count
    object_reads = 0
    traversed: set[EvidenceReference] = set()
    active: set[EvidenceReference] = set()

    async def resolve(
        reference: EvidenceReference,
        *,
        depth: int,
        require_sample: bool,
    ) -> SampleEvaluationRecord | None:
        nonlocal reference_count, object_reads
        reference_count += 1
        if reference_count > limits.max_sample_records:
            raise ValueError(
                "evidence reference count exceeds max_sample_records"
            )
        if depth > limits.max_reference_depth:
            raise ValueError(
                "evidence reference depth exceeds max_reference_depth"
            )
        if reference in active:
            raise ValueError("evidence reference graph contains a cycle")

        sample = resolved.get(reference)
        if sample is None:
            if isinstance(reference, BundleRecordReference):
                raise ValueError(
                    "nested bundle record reference is not closed by the bundle"
                )
            if object_store is None:
                raise ValueError(
                    "nested stored record validation requires object_store"
                )
            object_reads += 1
            if object_reads > limits.max_object_reads:
                raise ValueError("stored record reads exceed max_object_reads")
            decoded = await object_store.get(reference.reference)
            is_sample = (
                reference.reference.schema == SAMPLE_RECORD_OBJECT_SCHEMA
                and reference.schema_version
                == SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION
            )
            if is_sample:
                sample = SAMPLE_EVALUATION_RECORD_ADAPTER.validate_json(
                    canonical_json_bytes(decoded), strict=True
                )
                resolved[reference] = sample
            elif require_sample:
                raise ValueError(
                    "reused execution source must reference a sample evaluation record"
                )
        if require_sample and sample is None:
            raise ValueError(
                "reused execution source must resolve to sample evaluation evidence"
            )
        if sample is None or reference in traversed:
            return sample

        active.add(reference)
        try:
            await traverse(sample, depth=depth)
        finally:
            active.remove(reference)
        traversed.add(reference)
        return sample

    async def traverse(
        sample: SampleEvaluationRecord,
        *,
        depth: int,
    ) -> None:
        for reference, require_sample in _nested_evidence_references(sample):
            await resolve(
                reference,
                depth=depth + 1,
                require_sample=require_sample,
            )

    for sample in samples:
        await traverse(sample, depth=1)
    return object_reads


def _nested_evidence_references(
    sample: SampleEvaluationRecord,
) -> tuple[tuple[EvidenceReference, bool], ...]:
    references: list[tuple[EvidenceReference, bool]] = [
        (sample.sample.provenance.source_reference, False)
    ]
    if isinstance(sample, EvaluatedSampleRecord):
        references.extend(
            (execution.provenance.source_record, True)
            for execution in sample.executions
            if isinstance(execution.provenance, ReusedCandidateProvenance)
        )
    return tuple(references)


async def audit_evaluation_bundle(
    bundle_path: str | Path,
    /,
    *,
    object_store: ObjectStore | None,
    limits: EvaluationReadLimits,
) -> EvaluationBundleAudit:
    """Audit envelope integrity, then validate the complete domain graph."""

    reader = ArtifactBundleReader(bundle_path, limits=limits.bundle)
    manifest = await asyncio.to_thread(reader.audit)
    payload = EvaluationBundlePayload.model_validate_json(
        canonical_json_bytes(manifest.payload), strict=True
    )
    attempt = await asyncio.to_thread(_decode_attempt, reader, limits)
    if payload.attempt != attempt.identity:
        raise ValueError(
            "bundle payload attempt does not match attempt artifact"
        )
    if len(attempt.members) > limits.max_sample_records:
        raise ValueError("sample record count exceeds max_sample_records")

    expected = {EVALUATION_ATTEMPT_ARTIFACT}
    local_names = tuple(
        dict.fromkeys(
            member.record.artifact_name
            for member in attempt.members
            if isinstance(member.record, BundleRecordReference)
        )
    )
    for name in local_names:
        records = await asyncio.to_thread(
            _decode_sample_shard,
            reader,
            name,
            attempt=attempt.identity,
            limits=limits,
        )
        if len(records) > limits.max_sample_records:
            raise ValueError("sample shard count exceeds max_sample_records")
        expected.add(name)
        referenced = tuple(
            member.record
            for member in attempt.members
            if isinstance(member.record, BundleRecordReference)
            and member.record.artifact_name == name
        )
        if len(records) != len(referenced):
            raise ValueError("bundle sample shard has unreferenced records")
        for index, reference in enumerate(referenced):
            if (
                reference.schema != SAMPLE_RECORD_OBJECT_SCHEMA
                or reference.schema_version
                != SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION
            ):
                raise ValueError(
                    "bundle record reference schema is unsupported"
                )
            if reference.record_index != index:
                raise ValueError(
                    "bundle sample shard record indexes are not closed"
                )
            encoded = canonical_json_bytes(
                records[index].model_dump(mode="json")
            )
            if hashlib.sha256(encoded).hexdigest() != str(
                reference.record_sha256
            ):
                raise ValueError(
                    "bundle record reference hash does not match record"
                )
            member = next(
                candidate
                for candidate in attempt.members
                if candidate.record == reference
            )
            if (
                records[index].slot != member.slot
                or records[index].sample.identity != member.sample
            ):
                raise ValueError(
                    "sample record identity does not match attempt member"
                )

    stored = tuple(
        member.record
        for member in attempt.members
        if isinstance(member.record, StoredRecordReference)
    )
    if stored:
        if any(
            reference.reference.schema != SAMPLE_RECORD_OBJECT_SCHEMA
            or reference.schema_version
            != SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION
            for reference in stored
        ):
            raise ValueError("stored record reference schema is unsupported")
        index_names = tuple(
            descriptor.name
            for descriptor in manifest.artifacts
            if descriptor.name.startswith("sample-record-references-")
        )
        indexed: list[StoredRecordReference] = []
        for index, name in enumerate(index_names):
            if name != _sample_reference_artifact_name(index):
                raise ValueError(
                    "reference index shard names are not contiguous"
                )
            indexed.extend(
                await asyncio.to_thread(
                    _decode_reference_shard,
                    reader,
                    name,
                    attempt=attempt.identity,
                    limits=limits,
                )
            )
            if len(indexed) > limits.max_sample_records:
                raise ValueError("reference index exceeds max_sample_records")
            expected.add(name)
        if tuple(indexed) != stored:
            raise ValueError(
                "stored-record reference indexes do not close attempt"
            )

    if tuple(payload.projections) != tuple(
        EvaluationProjectionReference(
            kind=projection.kind,
            source_attempt=projection.source_attempt,
            artifact_name=projection.artifact_name,
        )
        for projection in payload.projections
    ):
        raise AssertionError("validated projections changed unexpectedly")
    for projection in payload.projections:
        if projection.source_attempt != attempt.identity:
            raise ValueError(
                "projection reference source attempt is incorrect"
            )
        expected_name = _PROJECTION_ARTIFACT_NAMES[projection.kind]
        if projection.artifact_name != expected_name:
            raise ValueError("projection reference artifact name is incorrect")
        header, _ = await asyncio.to_thread(
            _read_projection_artifact,
            reader,
            projection.kind,
            limits=limits,
        )
        if header.source_attempt != attempt.identity:
            raise ValueError("projection header source attempt is incorrect")
        expected.add(expected_name)

    declared = {descriptor.name for descriptor in manifest.artifacts}
    if declared != expected:
        raise ValueError("evaluation bundle artifact graph is not closed")
    samples, object_reads = await _restore_evaluation_records(
        reader,
        attempt,
        object_store=object_store,
        limits=limits,
    )
    validate_evaluation_attempt_graph(attempt, samples)
    return EvaluationBundleAudit(
        attempt=attempt.identity,
        artifact_count=len(manifest.artifacts),
        sample_record_count=sum(
            member.record is not None for member in attempt.members
        ),
        object_read_count=object_reads,
    )


__all__ = [
    "EVALUATION_BUNDLE_FORMAT",
    "EVALUATION_BUNDLE_SCHEMA_VERSION",
    "EVALUATION_PROJECTION_FORMAT",
    "EVALUATION_PROJECTION_SCHEMA_VERSION",
    "EvaluationBundleAudit",
    "EvaluationBundlePayload",
    "EvaluationReadLimits",
    "ProjectionArtifactHeader",
    "RestoredEvaluationAttempt",
    "SAMPLE_RECORD_OBJECT_SCHEMA",
    "audit_evaluation_bundle",
    "read_evaluation_projection",
    "restore_evaluation_attempt",
]
