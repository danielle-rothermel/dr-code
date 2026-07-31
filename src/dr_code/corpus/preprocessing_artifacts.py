"""Thin Parquet projections of preprocessing traces.

The projection is intentionally producer-blind with respect to source text:
candidate diagnostics and provenance are copied from the official trace.  No
candidate is parsed or compiled again at this persistence boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.atomic_directory import publish_staged_output_directory
from dr_code.corpus.durability import fsync_directory, fsync_file
from dr_code.corpus.preprocessing_contract import (
    PROJECTED_PART_SCHEMA_VERSION,
)
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.decoder_output import normalize_decoder_output
from dr_code.trace import (
    Absent,
    CodeCandidateSetArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
)

RESULTS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("decoder_output_presence", pa.string(), nullable=False),
        pa.field("raw_output_sha256", pa.string()),
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("outcome_code", pa.string()),
        pa.field("failure_code", pa.string()),
        pa.field("failed_step", pa.string()),
        pa.field("cause", pa.string()),
        pa.field("propagated_through", pa.list_(pa.string())),
        pa.field("final_candidate_count", pa.int64(), nullable=False),
    ]
)

_EXTRACTION_OPERATION_SCHEMA: Final = pa.struct(
    [
        pa.field("kind", pa.string(), nullable=False),
        pa.field("details_json", pa.string(), nullable=False),
    ]
)
ORIGIN_PATH_SCHEMA: Final = pa.struct(
    [pa.field("path", pa.list_(_EXTRACTION_OPERATION_SCHEMA), nullable=False)]
)
CANDIDATES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_index", pa.int64(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("cleaned_source", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("origins", pa.list_(ORIGIN_PATH_SCHEMA), nullable=False),
        pa.field("parse_ok", pa.bool_()),
        pa.field("parse_error", pa.string()),
        pa.field("compile_ok", pa.bool_()),
        pa.field("compile_error", pa.string()),
        pa.field("compile_warnings", pa.list_(pa.string())),
        pa.field("top_level_function_count", pa.int64()),
        pa.field("top_level_function_names", pa.list_(pa.string())),
        pa.field("top_level_async_function_names", pa.list_(pa.string())),
    ]
)
STEP_FACTS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("step_name", pa.string(), nullable=False),
        pa.field("facts_json", pa.string(), nullable=False),
    ]
)
REJECTIONS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("step_name", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string()),
        pa.field("input_index", pa.int64()),
        pa.field("reason_code", pa.string()),
        pa.field("details_json", pa.string(), nullable=False),
    ]
)
PROJECTED_ARTIFACT_SCHEMAS: Final[Mapping[str, pa.Schema]] = {
    "results": RESULTS_SCHEMA,
    "candidates": CANDIDATES_SCHEMA,
    "step_facts": STEP_FACTS_SCHEMA,
    "rejections": REJECTIONS_SCHEMA,
}

_RELATION_NAMES: Final = tuple(PROJECTED_ARTIFACT_SCHEMAS)
_ROW_GROUP_SIZE: Final = 65_536
_CANDIDATE_FACT_FIELDS: Final = (
    "parse_ok",
    "parse_error",
    "compile_ok",
    "compile_error",
    "compile_warnings",
    "top_level_function_count",
    "top_level_function_names",
    "top_level_async_function_names",
)
_REJECTION_COLUMNS: Final = frozenset(
    {
        "candidate_id",
        "input_index",
        "index",
        "reason_code",
        "reason",
    }
)
_OFFICIAL_PREPROCESSING_CONFIG: Final = (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize()
)
_OFFICIAL_PRODUCER: Final = TraceProducer(
    producer_id=_OFFICIAL_PREPROCESSING_CONFIG.definition_ref.definition_id,
    version=_OFFICIAL_PREPROCESSING_CONFIG.definition_ref.version,
    definition_hash=(
        _OFFICIAL_PREPROCESSING_CONFIG.definition_ref.identity_hash
    ),
    preprocessing_config_hash=(
        _OFFICIAL_PREPROCESSING_CONFIG.config_identity_hash
    ),
    implementation_hash=(_OFFICIAL_PREPROCESSING_CONFIG.implementation_hash),
)


@dataclass(slots=True)
class ProjectedArtifacts:
    """Rows in the four relations produced from one or more source samples."""

    results: list[dict[str, object]]
    candidates: list[dict[str, object]]
    step_facts: list[dict[str, object]]
    rejections: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class ProjectedPart:
    """One atomically published checkpoint shard."""

    part_id: str
    relation_paths: Mapping[str, Path]
    row_counts: Mapping[str, int]
    relation_sha256: Mapping[str, str]


class AtomicProjectedPartWriter:
    """Append bounded projections and atomically publish a complete shard."""

    def __init__(self, output_dir: str | Path, part_id: str) -> None:
        self.part_id = _validated_part_id(part_id)
        self._parts_dir = Path(output_dir) / "parts"
        self._part_dir = self._parts_dir / self.part_id
        self._temporary_dir: Path | None = None
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._counts = {name: 0 for name in _RELATION_NAMES}
        self._part: ProjectedPart | None = None
        self._aborted = False

    def __enter__(self) -> AtomicProjectedPartWriter:
        self._open()
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> bool:
        if exception_type is None:
            self.finish()
        else:
            self.abort()
        return False

    @property
    def part(self) -> ProjectedPart:
        if self._part is None:
            raise RuntimeError("projected part has not been published")
        return self._part

    def append(self, projected: ProjectedArtifacts) -> None:
        if not self._writers:
            raise RuntimeError("projected part writer is not open")
        for relation, rows in _relation_rows(projected):
            if not rows:
                continue
            table = pa.Table.from_pylist(
                rows, schema=PROJECTED_ARTIFACT_SCHEMAS[relation]
            )
            self._writers[relation].write_table(
                table, row_group_size=_ROW_GROUP_SIZE
            )
            self._counts[relation] += len(rows)

    def finish(self) -> ProjectedPart:
        if self._part is not None:
            return self._part
        if self._aborted:
            raise RuntimeError("cannot finish an aborted projected part")
        if not self._writers:
            raise RuntimeError("projected part writer is not open")
        self._close_writers()
        temporary_dir = self._require_temporary_dir()
        try:
            paths = {
                relation: temporary_dir / f"{relation}.parquet"
                for relation in _RELATION_NAMES
            }
            for path in paths.values():
                fsync_file(path)
            hashes = {name: file_sha256(path) for name, path in paths.items()}
            _atomic_write_json(
                temporary_dir / "manifest.json",
                {
                    "schema_version": PROJECTED_PART_SCHEMA_VERSION,
                    "part_id": self.part_id,
                    "relations": {
                        name: {
                            "rows": self._counts[name],
                            "sha256": hashes[name],
                        }
                        for name in _RELATION_NAMES
                    },
                },
            )
            fsync_directory(temporary_dir)
            publish_staged_output_directory(temporary_dir, self._part_dir)
        except FileExistsError:
            _authenticate_projected_part(
                self._part_dir,
                part_id=self.part_id,
                expected_counts=self._counts,
                expected_hashes=hashes,
            )
            self.abort()
            raise
        except BaseException:
            self.abort()
            raise
        fsync_directory(self._parts_dir)
        published_paths = {
            name: self._part_dir / f"{name}.parquet"
            for name in _RELATION_NAMES
        }
        self._part = ProjectedPart(
            part_id=self.part_id,
            relation_paths=published_paths,
            row_counts=dict(self._counts),
            relation_sha256=hashes,
        )
        return self._part

    def abort(self) -> None:
        if self._part is not None or self._aborted:
            return
        self._close_writers()
        if self._temporary_dir is not None and self._temporary_dir.exists():
            shutil.rmtree(self._temporary_dir)
        self._aborted = True

    def _open(self) -> None:
        if self._part is not None or self._aborted or self._writers:
            raise RuntimeError("projected part writer cannot be reopened")
        if self._part_dir.exists():
            _authenticate_projected_part(
                self._part_dir,
                part_id=self.part_id,
            )
            raise FileExistsError(
                f"projected part already exists: {self._part_dir}"
            )
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        fsync_directory(self._parts_dir.parent)
        self._temporary_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.part_id}.",
                suffix=".tmp",
                dir=self._parts_dir,
            )
        )
        try:
            for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
                self._writers[relation] = pq.ParquetWriter(
                    self._temporary_dir / f"{relation}.parquet",
                    schema,
                    compression="zstd",
                    use_dictionary=False,
                    write_statistics=True,
                    version="2.6",
                )
        except BaseException:
            self.abort()
            raise

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def _require_temporary_dir(self) -> Path:
        if self._temporary_dir is None:
            raise RuntimeError(
                "projected part writer has no staging directory"
            )
        return self._temporary_dir


def project_preprocessing_result(
    sample_id: str,
    decoder_output: str | None,
    trace: Trace | None,
) -> ProjectedArtifacts:
    """Project one source record while preserving missing versus present."""

    if not sample_id:
        raise ValueError("sample_id must be non-empty")
    if decoder_output is None:
        if trace is not None:
            raise ValueError("missing decoder output must not have a trace")
        return ProjectedArtifacts(
            results=[
                {
                    "sample_id": sample_id,
                    "decoder_output_presence": "missing",
                    "raw_output_sha256": None,
                    "outcome": "decoder_output_missing",
                    "outcome_code": None,
                    "failure_code": None,
                    "failed_step": None,
                    "cause": None,
                    "propagated_through": None,
                    "final_candidate_count": 0,
                }
            ],
            candidates=[],
            step_facts=[],
            rejections=[],
        )
    if trace is None:
        raise ValueError(
            "present decoder output requires a preprocessing trace"
        )
    trace_input = trace.value("input")
    if not isinstance(trace_input, TextArtifact):
        raise ValueError("preprocessing trace input must be a TextArtifact")
    normalized_input = normalize_decoder_output(decoder_output)
    if trace_input.text != normalized_input.text:
        raise ValueError(
            "preprocessing trace input does not match decoder output"
        )
    if trace.producer != _OFFICIAL_PRODUCER:
        raise ValueError(
            "preprocessing trace was not produced by the official definition"
        )
    output = trace.value("output")
    if not isinstance(output, Absent | CodeCandidateSetArtifact):
        raise ValueError(
            "preprocessing trace output must be Absent or "
            "CodeCandidateSetArtifact"
        )
    candidates = (
        _candidate_rows(sample_id, output, trace)
        if isinstance(output, CodeCandidateSetArtifact)
        else []
    )
    return ProjectedArtifacts(
        results=[_result_row(sample_id, decoder_output, output, trace)],
        candidates=candidates,
        step_facts=_step_fact_rows(sample_id, trace),
        rejections=_rejection_rows(sample_id, trace),
    )


def write_projected_part(
    output_dir: str | Path,
    part_id: str,
    projected: ProjectedArtifacts,
) -> ProjectedPart:
    with AtomicProjectedPartWriter(output_dir, part_id) as writer:
        writer.append(projected)
    return writer.part


def combine_projected_parts(
    output_dir: str | Path, part_ids: Iterable[str]
) -> Mapping[str, Path]:
    """Stream checkpoint parts into atomically replaced final relations."""

    normalized = tuple(_validated_part_id(value) for value in part_ids)
    root = Path(output_dir)
    result: dict[str, Path] = {}
    for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        destination = root / f"{relation}.parquet"
        temporary = root / f".{relation}.parquet.tmp"
        try:
            with pq.ParquetWriter(
                temporary,
                schema,
                compression="zstd",
                use_dictionary=False,
                write_statistics=True,
                version="2.6",
            ) as writer:
                for part_id in normalized:
                    source = root / "parts" / part_id / f"{relation}.parquet"
                    parquet = pq.ParquetFile(source)
                    if not parquet.schema_arrow.equals(schema):
                        raise ValueError(
                            f"unexpected schema in projected part: {source}"
                        )
                    for batch in parquet.iter_batches(
                        batch_size=_ROW_GROUP_SIZE, columns=schema.names
                    ):
                        writer.write_batch(batch)
            fsync_file(temporary)
            os.replace(temporary, destination)
            fsync_directory(root)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        result[relation] = destination
    return result


def validate_origin_paths(value: object) -> list[dict[str, object]]:
    """Validate schema-2 ordered provenance paths and decode details JSON."""

    if not isinstance(value, list) or not value:
        raise ValueError("candidate origins must be a non-empty list")
    normalized: list[dict[str, object]] = []
    for origin in value:
        if not isinstance(origin, Mapping):
            raise ValueError("candidate origin must be an object")
        raw_path = origin.get("path")
        if not isinstance(raw_path, list) or not raw_path:
            raise ValueError("candidate origin path must be non-empty")
        path: list[dict[str, object]] = []
        for operation in raw_path:
            if not isinstance(operation, Mapping):
                raise ValueError("extraction operation must be an object")
            kind = operation.get("kind")
            details_json = operation.get("details_json")
            if not isinstance(kind, str) or not kind:
                raise ValueError("origin operation name must be non-empty")
            if not isinstance(details_json, str):
                raise ValueError(
                    "extraction operation details_json must be a string"
                )
            details = _canonical_json_object(
                details_json,
                label="extraction operation details_json",
            )
            path.append({"kind": kind, "details": details})
        normalized.append({"path": path})
    return normalized


class _DuplicateJsonKeyError(ValueError):
    pass


def _canonical_json_object(value: str, *, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJsonKeyError, ValueError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must encode an object")
    if value != _canonical_json(decoded):
        raise ValueError(f"{label} is not canonical JSON")
    return decoded


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def read_part_manifest(part_dir: Path) -> dict[str, object]:
    try:
        value = json.loads((part_dir / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read part manifest: {part_dir}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"part manifest must be an object: {part_dir}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_row(
    sample_id: str,
    decoder_output: str,
    output: Absent | CodeCandidateSetArtifact,
    trace: Trace,
) -> dict[str, object]:
    if isinstance(output, Absent):
        return {
            "sample_id": sample_id,
            "decoder_output_presence": "present",
            "raw_output_sha256": _sha256_text(decoder_output),
            "outcome": output.failure_code,
            "outcome_code": None,
            "failure_code": output.failure_code,
            "failed_step": output.failed_step,
            "cause": output.cause,
            "propagated_through": list(output.propagated_through),
            "final_candidate_count": 0,
        }
    outcome_code = _outcome_code(trace)
    if outcome_code is None:
        raise ValueError("successful preprocessing trace has no outcome_code")
    return {
        "sample_id": sample_id,
        "decoder_output_presence": "present",
        "raw_output_sha256": _sha256_text(decoder_output),
        "outcome": outcome_code,
        "outcome_code": outcome_code,
        "failure_code": None,
        "failed_step": None,
        "cause": None,
        "propagated_through": None,
        "final_candidate_count": len(output.candidates),
    }


def _outcome_code(trace: Trace) -> str | None:
    outcome: str | None = None
    for facts in trace.step_facts.values():
        value = facts.get("outcome_code")
        if isinstance(value, str):
            outcome = value
    return outcome


def _step_fact_rows(sample_id: str, trace: Trace) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "step_name": step_name,
            "facts_json": _canonical_json(facts),
        }
        for step_name, facts in trace.step_facts.items()
    ]


def _rejection_rows(sample_id: str, trace: Trace) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for step_name, facts in trace.step_facts.items():
        rejections = facts.get("rejections")
        if not isinstance(rejections, list):
            continue
        for rejection in rejections:
            if not isinstance(rejection, Mapping):
                continue
            input_index = rejection.get("input_index", rejection.get("index"))
            reason_code = rejection.get("reason_code", rejection.get("reason"))
            result.append(
                {
                    "sample_id": sample_id,
                    "step_name": step_name,
                    "candidate_id": _string_or_none(
                        rejection.get("candidate_id")
                    ),
                    "input_index": _int_or_none(input_index),
                    "reason_code": _string_or_none(reason_code),
                    "details_json": _canonical_json(
                        {
                            key: value
                            for key, value in rejection.items()
                            if key not in _REJECTION_COLUMNS
                        }
                    ),
                }
            )
    return result


def _candidate_rows(
    sample_id: str,
    output: CodeCandidateSetArtifact,
    trace: Trace,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (source, lineage) in enumerate(
        zip(output.candidates, output.lineage, strict=True)
    ):
        candidate_id = lineage.candidate_id
        if candidate_id is None:
            raise ValueError(
                "final preprocessing candidate is missing candidate_id"
            )
        rows.append(
            {
                "sample_id": sample_id,
                "candidate_index": index,
                "candidate_id": candidate_id,
                "cleaned_source": source,
                "source_sha256": _sha256_text(source),
                "origins": [
                    {
                        "path": [
                            {
                                "kind": operation.kind,
                                "details_json": _canonical_json(
                                    operation.details
                                ),
                            }
                            for operation in origin.path
                        ]
                    }
                    for origin in lineage.origins
                ],
                **_candidate_diagnostics(candidate_id, trace),
            }
        )
    return rows


def _candidate_diagnostics(
    candidate_id: str, trace: Trace
) -> dict[str, object]:
    result: dict[str, object] = {
        field: None for field in _CANDIDATE_FACT_FIELDS
    }
    for facts in trace.step_facts.values():
        survivors = facts.get("survivors")
        if not isinstance(survivors, list):
            continue
        for survivor in survivors:
            if (
                isinstance(survivor, Mapping)
                and survivor.get("candidate_id") == candidate_id
            ):
                for field in _CANDIDATE_FACT_FIELDS:
                    if field in survivor:
                        result[field] = survivor[field]
    return result


def _relation_rows(
    value: ProjectedArtifacts,
) -> tuple[tuple[str, list[dict[str, object]]], ...]:
    return (
        ("results", value.results),
        ("candidates", value.candidates),
        ("step_facts", value.step_facts),
        ("rejections", value.rejections),
    )


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def _authenticate_projected_part(
    part_dir: Path,
    *,
    part_id: str,
    expected_counts: Mapping[str, int] | None = None,
    expected_hashes: Mapping[str, str] | None = None,
) -> None:
    """Authenticate an existing immutable part before rejecting publication."""

    if part_dir.is_symlink() or not part_dir.is_dir():
        raise ValueError(
            f"projected part is not a regular directory: {part_dir}"
        )
    expected_names = {
        "manifest.json",
        *(f"{relation}.parquet" for relation in _RELATION_NAMES),
    }
    if {path.name for path in part_dir.iterdir()} != expected_names:
        raise ValueError(
            f"projected part contains unexpected files: {part_dir}"
        )
    manifest = read_part_manifest(part_dir)
    if (
        manifest.get("schema_version") != PROJECTED_PART_SCHEMA_VERSION
        or manifest.get("part_id") != part_id
    ):
        raise ValueError(f"projected part manifest is invalid: {part_dir}")
    relations = manifest.get("relations")
    if not isinstance(relations, dict) or set(relations) != set(
        _RELATION_NAMES
    ):
        raise ValueError(f"projected part relations are invalid: {part_dir}")
    for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        path = part_dir / f"{relation}.parquet"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"projected part relation is invalid: {path}")
        parquet = pq.ParquetFile(path)
        metadata = relations.get(relation)
        if not isinstance(metadata, dict):
            raise ValueError(f"projected part metadata is invalid: {path}")
        actual_hash = file_sha256(path)
        if (
            not parquet.schema_arrow.equals(schema)
            or metadata.get("rows") != parquet.metadata.num_rows
            or metadata.get("sha256") != actual_hash
            or (
                expected_counts is not None
                and expected_counts[relation] != parquet.metadata.num_rows
            )
            or (
                expected_hashes is not None
                and expected_hashes[relation] != actual_hash
            )
        ):
            raise ValueError(f"projected part authentication failed: {path}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _validated_part_id(part_id: str) -> str:
    if (
        not part_id
        or part_id in {".", ".."}
        or Path(part_id).name != part_id
        or part_id.startswith(".")
        or part_id.endswith(".tmp")
    ):
        raise ValueError("part_id must be a single non-empty path component")
    return part_id


__all__ = [
    "AtomicProjectedPartWriter",
    "CANDIDATES_SCHEMA",
    "ORIGIN_PATH_SCHEMA",
    "PROJECTED_ARTIFACT_SCHEMAS",
    "ProjectedArtifacts",
    "ProjectedPart",
    "REJECTIONS_SCHEMA",
    "RESULTS_SCHEMA",
    "STEP_FACTS_SCHEMA",
    "combine_projected_parts",
    "file_sha256",
    "project_preprocessing_result",
    "read_part_manifest",
    "validate_origin_paths",
    "write_projected_part",
]
