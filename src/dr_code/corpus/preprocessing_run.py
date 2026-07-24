"""Row-group-checkpointed preprocessing corpus persistence."""

from __future__ import annotations

import hashlib
import errno
import fcntl
import json
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError, model_validator

from dr_code.corpus.preprocessing_artifacts import (
    AtomicProjectedPartWriter,
    PROJECTED_ARTIFACT_SCHEMAS,
    ProjectedArtifacts,
    combine_projected_parts,
    file_sha256,
    project_preprocessing_result,
    read_part_manifest,
    validate_origin_paths,
)
from dr_code.corpus.durability import fsync_directory, fsync_file
from dr_code.corpus.coordinate_validation import (
    CoordinateValidationError,
    validate_preprocessing_coordinates,
)
from dr_code.corpus.output_paths import (
    UnsafeOutputPathError,
    validate_output_path,
    validate_owned_tree,
    validate_reserved_path,
)
from dr_code.corpus.preprocessing_contract import (
    PREPROCESSING_MANIFEST_SCHEMA_VERSION,
    PROJECTED_PART_SCHEMA_VERSION,
)
from dr_code.corpus.runtime_provenance import installed_environment_provenance
from dr_code.corpus.stable_files import StableFile, stable_file
from dr_code.eval.lifecycle import PreprocessingConfig
from dr_code.implementation_identity import package_source_digest
from dr_code.models import FrozenModel
from dr_code.preprocessing.candidate_identity import candidate_id_for_source
from dr_code.preprocessing.decoder_output import normalize_decoder_output
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.runner import (
    BoundPreprocessingRunner,
    bind_preprocessing,
)
from dr_code.trace import CandidateOrigin, TextArtifact

RELATION_NAMES: Final = tuple(PROJECTED_ARTIFACT_SCHEMAS)
REQUIRED_INPUT_COLUMNS: Final = ("sample_id", "decoder_output")
_BATCH_ROW_LIMIT: Final = 65_536
_PROCESS_WRITER_GUARD: Final = threading.Lock()
_PROCESS_WRITER_LOCKS: Final[set[Path]] = set()


class _CandidateCounts(FrozenModel):
    input_candidate_count: int = Field(ge=0)
    output_candidate_count: int = Field(ge=0)


class _RequireNonblankFacts(FrozenModel):
    text_character_count: int = Field(ge=0)
    is_nonblank: bool


class _DecoderOutputValidationFacts(FrozenModel):
    text_character_count: int = Field(ge=0)
    contains_nul: bool
    contains_surrogate: bool


class _OperationCount(FrozenModel):
    kind: str = Field(min_length=1)
    count: int = Field(ge=0)


class _ExtractFacts(FrozenModel):
    candidate_count: int = Field(ge=0)
    operation_counts: tuple[_OperationCount, ...]
    paths: tuple[CandidateOrigin, ...]

    @model_validator(mode="after")
    def validate_path_count(self) -> _ExtractFacts:
        if len(self.paths) != self.candidate_count:
            raise ValueError("candidate_count does not match paths")
        actual_counts = Counter(
            operation.kind
            for origin in self.paths
            for operation in origin.path
        )
        recorded_counts = {
            item.kind: item.count for item in self.operation_counts
        }
        if (
            len(recorded_counts) != len(self.operation_counts)
            or actual_counts != recorded_counts
        ):
            raise ValueError("operation_counts do not match candidate paths")
        return self


class _SalvageRepair(FrozenModel):
    input_index: int = Field(ge=0)
    output_index: int = Field(ge=0)


class _SalvageFacts(_CandidateCounts):
    salvage_candidate_count: int = Field(ge=0)
    repairs: tuple[_SalvageRepair, ...]

    @model_validator(mode="after")
    def validate_repairs(self) -> _SalvageFacts:
        if self.salvage_candidate_count != len(self.repairs):
            raise ValueError("salvage_candidate_count does not match repairs")
        if self.output_candidate_count != (
            self.input_candidate_count + self.salvage_candidate_count
        ):
            raise ValueError("salvage candidate counts do not reconcile")
        return self


class _NonblankRejection(FrozenModel):
    index: int = Field(ge=0)
    reason: Literal["blank_or_whitespace"]


class _NonblankFacts(_CandidateCounts):
    rejections: tuple[_NonblankRejection, ...]

    @model_validator(mode="after")
    def validate_rejections(self) -> _NonblankFacts:
        if self.output_candidate_count + len(self.rejections) != (
            self.input_candidate_count
        ):
            raise ValueError("nonblank candidate counts do not reconcile")
        indexes = [item.index for item in self.rejections]
        if len(indexes) != len(set(indexes)) or any(
            index >= self.input_candidate_count for index in indexes
        ):
            raise ValueError("nonblank rejection membership is invalid")
        return self


class _Transformation(FrozenModel):
    kind: Literal["infer_missing_imports", "lambda_to_function"]
    input_source: str
    output_source: str


class _Inspection(FrozenModel):
    candidate_id: str = Field(min_length=1)
    parse_ok: bool
    parse_error: str | None
    compile_ok: bool
    compile_error: str | None
    compile_warnings: tuple[str, ...]
    top_level_function_count: int = Field(ge=0)
    top_level_function_names: tuple[str, ...]
    top_level_async_function_names: tuple[str, ...]

    @model_validator(mode="after")
    def validate_diagnostics(self) -> _Inspection:
        if self.parse_ok != (self.parse_error is None):
            raise ValueError("parse diagnostics are inconsistent")
        if self.compile_ok != (self.compile_error is None):
            raise ValueError("compile diagnostics are inconsistent")
        if self.top_level_function_count != len(self.top_level_function_names):
            raise ValueError("top-level function count does not reconcile")
        if not set(self.top_level_async_function_names).issubset(
            self.top_level_function_names
        ):
            raise ValueError("async function names are not top-level names")
        return self


class _IdentifyFailureFacts(FrozenModel):
    input_candidate_count: Literal[0]


class _IdentifyFacts(FrozenModel):
    input_candidate_count: int = Field(ge=1)
    unique_input_source_count: int = Field(ge=1)
    identified_candidate_count: int = Field(ge=1)
    inspection_count: int = Field(ge=1)
    transformations: tuple[_Transformation, ...]
    inspections: tuple[_Inspection, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> _IdentifyFacts:
        if self.identified_candidate_count != len(self.inspections):
            raise ValueError(
                "identified_candidate_count does not match inspections"
            )
        candidate_ids = [item.candidate_id for item in self.inspections]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("identified candidate membership is not unique")
        if self.unique_input_source_count > self.input_candidate_count:
            raise ValueError(
                "unique input count exceeds input candidate count"
            )
        if self.inspection_count < self.identified_candidate_count:
            raise ValueError(
                "inspection_count is below identified candidate count"
            )
        return self


class _CandidateCoordinate(FrozenModel):
    input_index: int = Field(ge=0)
    candidate_id: str = Field(min_length=1)


class _CandidateDiagnostics(_CandidateCoordinate):
    parse_ok: bool
    parse_error: str | None
    compile_ok: bool
    compile_error: str | None
    compile_warnings: tuple[str, ...]

    @model_validator(mode="after")
    def validate_diagnostics(self) -> _CandidateDiagnostics:
        if self.parse_ok != (self.parse_error is None):
            raise ValueError("parse diagnostics are inconsistent")
        if self.compile_ok != (self.compile_error is None):
            raise ValueError("compile diagnostics are inconsistent")
        return self


class _RejectedDiagnostics(_CandidateDiagnostics):
    reason_code: str = Field(min_length=1)


class _FunctionDiagnostics(_CandidateDiagnostics):
    top_level_function_count: int = Field(ge=0)
    top_level_function_names: tuple[str, ...]
    top_level_async_function_names: tuple[str, ...]
    has_async_top_level_function: bool

    @model_validator(mode="after")
    def validate_functions(self) -> _FunctionDiagnostics:
        if self.top_level_function_count != len(self.top_level_function_names):
            raise ValueError("top-level function count does not reconcile")
        if not set(self.top_level_async_function_names).issubset(
            self.top_level_function_names
        ):
            raise ValueError("async function names are not top-level names")
        if self.has_async_top_level_function != bool(
            self.top_level_async_function_names
        ):
            raise ValueError("async top-level function flag is inconsistent")
        return self


class _RejectedFunctionDiagnostics(_FunctionDiagnostics):
    reason_code: str = Field(min_length=1)


class _SimpleFilterFacts(FrozenModel):
    input_candidate_count: int = Field(ge=0)
    survivor_candidate_count: int = Field(ge=0)
    survivors: tuple[_CandidateCoordinate, ...]
    rejections: tuple[_RejectedDiagnostics, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> _SimpleFilterFacts:
        _validate_filter_counts(
            self.input_candidate_count,
            self.survivor_candidate_count,
            self.survivors,
            self.rejections,
        )
        return self


class _DiagnosticFilterFacts(FrozenModel):
    input_candidate_count: int = Field(ge=0)
    survivor_candidate_count: int = Field(ge=0)
    survivors: tuple[_CandidateDiagnostics, ...]
    rejections: tuple[_RejectedDiagnostics, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> _DiagnosticFilterFacts:
        _validate_filter_counts(
            self.input_candidate_count,
            self.survivor_candidate_count,
            self.survivors,
            self.rejections,
        )
        return self


class _FunctionFilterFacts(FrozenModel):
    input_candidate_count: int = Field(ge=0)
    survivor_candidate_count: int = Field(ge=0)
    survivors: tuple[_FunctionDiagnostics, ...]
    rejections: tuple[_RejectedDiagnostics | _RejectedFunctionDiagnostics, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> _FunctionFilterFacts:
        _validate_filter_counts(
            self.input_candidate_count,
            self.survivor_candidate_count,
            self.survivors,
            self.rejections,
        )
        return self


class _CandidateCount(FrozenModel):
    candidate_count: int = Field(ge=0)


class _SuccessfulReturnFacts(FrozenModel):
    outcome_code: Literal["function_candidates_extracted"]
    candidate_count: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class _ParsedStepFact:
    raw: dict[str, object]
    value: FrozenModel


def _validate_filter_counts(
    input_count: int,
    survivor_count: int,
    survivors: tuple[object, ...],
    rejections: tuple[object, ...],
) -> None:
    if survivor_count != len(survivors):
        raise ValueError("survivor_candidate_count does not match survivors")
    if input_count != survivor_count + len(rejections):
        raise ValueError("filter candidate counts do not reconcile")
    indexes = [
        item.input_index
        for item in (*survivors, *rejections)
        if isinstance(item, _CandidateCoordinate)
    ]
    if sorted(indexes) != list(range(input_count)):
        raise ValueError("filter candidate membership does not reconcile")


_FACT_SCHEMAS: Final[Mapping[str, tuple[type[FrozenModel], ...]]] = {
    "validate_decoder_output": (_DecoderOutputValidationFacts,),
    "require_nonblank_text": (_RequireNonblankFacts,),
    "extract_candidates": (_ExtractFacts,),
    "strip_fences": (_CandidateCounts,),
    "dedent": (_CandidateCounts,),
    "normalize_smart_quotes": (_CandidateCounts,),
    "split_on_name_guard": (_CandidateCounts,),
    "expand_last_return_salvage": (_SalvageFacts,),
    "repair_import_lines": (_CandidateCounts,),
    "dedupe_imports": (_CandidateCounts,),
    "filter_nonblank_candidates": (_NonblankFacts,),
    "identify_candidates": (_IdentifyFailureFacts, _IdentifyFacts),
    "filter_plain_literal": (_SimpleFilterFacts,),
    "filter_code_repr": (_SimpleFilterFacts,),
    "filter_compilable": (_DiagnosticFilterFacts,),
    "filter_has_top_level_function": (_FunctionFilterFacts,),
    "materialize_candidates": (_CandidateCount,),
    "return_all": (_CandidateCount, _SuccessfulReturnFacts),
}
_PIPELINE_STEPS: Final = tuple(
    step.instance_name
    for step in HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.steps
)
_STEP_POSITION: Final = {
    step_name: index for index, step_name in enumerate(_PIPELINE_STEPS)
}
_FACT_POSITION: Final = {
    step_name: index
    for index, step_name in enumerate(
        ("validate_decoder_output", *_PIPELINE_STEPS)
    )
}
_FAILURE_BY_STEP: Final = {
    "validate_decoder_output": PreprocessingFailureCode.DECODER_OUTPUT_INVALID,
    "require_nonblank_text": PreprocessingFailureCode.DECODER_OUTPUT_BLANK,
    "extract_candidates": PreprocessingFailureCode.NO_CODE_CANDIDATES,
    "filter_nonblank_candidates": (
        PreprocessingFailureCode.NO_NONBLANK_CLEANED_CANDIDATE
    ),
    "identify_candidates": PreprocessingFailureCode.NO_CANDIDATES_TO_IDENTIFY,
    "filter_plain_literal": PreprocessingFailureCode.PLAIN_LITERAL_ONLY,
    "filter_code_repr": PreprocessingFailureCode.CODE_REPR_ONLY,
    "filter_compilable": PreprocessingFailureCode.NO_COMPILABLE_CANDIDATE,
    "filter_has_top_level_function": (
        PreprocessingFailureCode.NO_TOP_LEVEL_FUNCTION_CANDIDATE
    ),
    "return_all": PreprocessingFailureCode.NO_CANDIDATES_TO_RETURN,
}


class CorpusRunError(ValueError):
    """Input, checkpoint state, or output violates the corpus contract."""


def run_preprocessing_corpus(
    *,
    input_path: Path | str,
    output_root: Path | str,
    run_id: str | None = None,
    batch_size: int = 1_000,
    max_row_groups: int | None = None,
) -> Path:
    """Write or resume one immutable preprocessing run.

    Checkpoints live under ``<run_id>.partial/parts``.  A completed artifact
    becomes visible only when the fully validated partial directory is renamed
    to ``<run_id>``.
    """

    if batch_size < 1:
        raise CorpusRunError("batch_size must be at least 1")
    if max_row_groups is not None and max_row_groups < 1:
        raise CorpusRunError("max_row_groups must be at least 1 when set")
    if run_id is not None:
        _validate_run_id(run_id)
    requested_output_root = Path(output_root).expanduser()
    try:
        validate_output_path(
            requested_output_root,
            label="preprocessing output root",
        )
    except UnsafeOutputPathError as exc:
        raise CorpusRunError(str(exc)) from exc
    source_path = Path(input_path).expanduser().resolve(strict=True)
    with stable_file(source_path, label="preprocessing input") as snapshot:
        return _run_preprocessing_snapshot(
            input_snapshot=snapshot,
            output_root=output_root,
            run_id=run_id,
            batch_size=batch_size,
            max_row_groups=max_row_groups,
        )


def _run_preprocessing_snapshot(
    *,
    input_snapshot: StableFile,
    output_root: Path | str,
    run_id: str | None,
    batch_size: int,
    max_row_groups: int | None,
) -> Path:
    """Process only the bytes captured before any output mutation."""

    try:
        root = validate_output_path(
            output_root,
            label="preprocessing output root",
        )
    except UnsafeOutputPathError as exc:
        raise CorpusRunError(str(exc)) from exc
    source = pq.ParquetFile(input_snapshot.path)
    _validate_input_schema(source.schema_arrow)
    input_coordinates = _input_coordinates(input_snapshot, source)
    resolved_run_id = run_id or _generated_run_id(input_coordinates)
    _validate_run_id(resolved_run_id)
    completed_dir = root / resolved_run_id
    partial_dir = root / f"{resolved_run_id}.partial"
    _validate_run_output_namespace(
        root=root,
        completed_dir=completed_dir,
        partial_dir=partial_dir,
        run_id=resolved_run_id,
    )

    preprocessing_config = (
        HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION.materialize()
    )
    runner = bind_preprocessing(preprocessing_config)
    immutable = _immutable_coordinates(
        run_id=resolved_run_id,
        input_coordinates=input_coordinates,
        batch_size=batch_size,
        preprocessing_config=preprocessing_config,
    )
    _validate_existing_manifest_compatibility(partial_dir, immutable)
    root.mkdir(parents=True, exist_ok=True)
    with _single_writer_guard(root, resolved_run_id):
        if completed_dir.exists():
            raise FileExistsError(
                f"completed run already exists: {completed_dir}"
            )
        manifest = _load_or_create_manifest(partial_dir, immutable)
        completed = _completed_row_groups(
            manifest, expected_count=source.num_row_groups
        )
        _validate_completed_parts(partial_dir, completed)
        if manifest.get("complete") is True:
            _publish_validated_complete_partial(
                partial_dir=partial_dir,
                completed_dir=completed_dir,
                manifest=manifest,
                input_parquet=source,
                completed=completed,
            )
            return completed_dir

        processed = 0
        for row_group_index in range(source.num_row_groups):
            if row_group_index in completed:
                continue
            if max_row_groups is not None and processed >= max_row_groups:
                break
            part_id = _part_id(row_group_index)
            _remove_orphan_part(partial_dir, part_id)
            _write_row_group(
                parquet=source,
                row_group_index=row_group_index,
                batch_size=batch_size,
                runner=runner,
                partial_dir=partial_dir,
                part_id=part_id,
            )
            completed.add(row_group_index)
            manifest["completed_row_groups"] = sorted(completed)
            manifest["relation_totals"] = _part_relation_totals(
                partial_dir, completed
            )
            manifest["outcome_totals"] = _part_outcome_totals(
                partial_dir, completed
            )
            manifest["updated_at"] = _timestamp()
            _write_manifest(partial_dir, manifest)
            processed += 1

        if len(completed) != source.num_row_groups:
            return partial_dir

        relation_paths = combine_projected_parts(
            partial_dir, [_part_id(index) for index in sorted(completed)]
        )
        totals = _validate_completed_relations(
            relation_paths=relation_paths,
            input_parquet=source,
            expected_rows=input_coordinates["expected_rows"],
        )
        manifest.update(totals)
        manifest["relation_sha256"] = {
            relation: file_sha256(path)
            for relation, path in relation_paths.items()
        }
        manifest["complete"] = True
        manifest["completed_at"] = _timestamp()
        manifest["updated_at"] = manifest["completed_at"]
        _write_manifest(partial_dir, manifest)
        if completed_dir.exists():
            raise FileExistsError(
                f"completed run already exists: {completed_dir}"
            )
        os.replace(partial_dir, completed_dir)
        fsync_directory(root)
        return completed_dir


@contextmanager
def _single_writer_guard(root: Path, run_id: str) -> Iterator[None]:
    lock_path = (root / f".{run_id}.lock").resolve()
    with _PROCESS_WRITER_GUARD:
        if lock_path in _PROCESS_WRITER_LOCKS:
            raise CorpusRunError(
                f"preprocessing run is owned by a live writer: {run_id!r}"
            )
        _PROCESS_WRITER_LOCKS.add(lock_path)

    stream = None
    acquired = False
    try:
        stream = lock_path.open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise CorpusRunError(
                f"preprocessing run is owned by a live writer: {run_id!r}"
            ) from exc
        yield
    finally:
        if stream is not None:
            try:
                if acquired:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        with _PROCESS_WRITER_GUARD:
            _PROCESS_WRITER_LOCKS.remove(lock_path)


def _write_row_group(
    *,
    parquet: pq.ParquetFile,
    row_group_index: int,
    batch_size: int,
    runner: BoundPreprocessingRunner,
    partial_dir: Path,
    part_id: str,
) -> None:
    with AtomicProjectedPartWriter(partial_dir, part_id) as writer:
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group_index],
            columns=list(REQUIRED_INPUT_COLUMNS),
        ):
            writer.append(_project_batch(batch, runner))


def _project_batch(
    batch: pa.RecordBatch, runner: BoundPreprocessingRunner
) -> ProjectedArtifacts:
    result = ProjectedArtifacts([], [], [], [])
    for sample_id, decoder_output in zip(
        batch.column("sample_id").to_pylist(),
        batch.column("decoder_output").to_pylist(),
        strict=True,
    ):
        if not isinstance(sample_id, str):
            raise CorpusRunError("sample_id must be a string")
        if decoder_output is not None and not isinstance(decoder_output, str):
            raise CorpusRunError("decoder_output must be a string or null")
        trace = (
            None
            if decoder_output is None
            else runner.run(TextArtifact(text=decoder_output))
        )
        projected = project_preprocessing_result(
            sample_id, decoder_output, trace
        )
        result.results.extend(projected.results)
        result.candidates.extend(projected.candidates)
        result.step_facts.extend(projected.step_facts)
        result.rejections.extend(projected.rejections)
    return result


def _validate_input_schema(schema: pa.Schema) -> None:
    missing = [
        name for name in REQUIRED_INPUT_COLUMNS if name not in schema.names
    ]
    if missing:
        raise CorpusRunError(
            "input Parquet is missing required column(s): "
            + ", ".join(missing)
        )
    for name in REQUIRED_INPUT_COLUMNS:
        if not pa.types.is_string(schema.field(name).type):
            raise CorpusRunError(
                f"input column {name!r} must have Arrow string type"
            )
    if schema.field("sample_id").nullable:
        raise CorpusRunError("input column 'sample_id' must be non-nullable")


def _input_coordinates(
    snapshot: StableFile, parquet: pq.ParquetFile
) -> dict[str, object]:
    return {
        "path": str(snapshot.source_path),
        "sha256": snapshot.sha256,
        "size": snapshot.size,
        "schema_hex": parquet.schema_arrow.serialize().to_pybytes().hex(),
        "expected_rows": parquet.metadata.num_rows,
        "expected_row_groups": parquet.num_row_groups,
        "row_groups": [
            {
                "index": index,
                "rows": parquet.metadata.row_group(index).num_rows,
                "total_byte_size": (
                    parquet.metadata.row_group(index).total_byte_size
                ),
            }
            for index in range(parquet.num_row_groups)
        ],
    }


def _immutable_coordinates(
    *,
    run_id: str,
    input_coordinates: Mapping[str, object],
    batch_size: int,
    preprocessing_config: PreprocessingConfig,
) -> dict[str, object]:
    coordinates = {
        "schema_version": PREPROCESSING_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "input": dict(input_coordinates),
        "preprocessing_definition_ref": (
            preprocessing_config.definition_ref.model_dump(mode="json")
        ),
        "preprocessing_config": preprocessing_config.model_dump(mode="json"),
        "preprocessing_definition_identity": (
            preprocessing_config.definition_ref.identity_hash
        ),
        "preprocessing_config_identity": (
            preprocessing_config.config_identity_hash
        ),
        "resolved_step_versions": [
            {
                "instance_name": instance_name,
                "step": step,
                "version": version,
                "implementation_hash": implementation_hash,
            }
            for (
                instance_name,
                step,
                version,
                implementation_hash,
            ) in preprocessing_config.resolved_step_versions
        ],
        "source": _source_coordinates(),
        "installed_environment": installed_environment_provenance(),
        "batch_size": batch_size,
    }
    try:
        validate_preprocessing_coordinates(coordinates)
    except CoordinateValidationError as exc:
        raise CorpusRunError(str(exc)) from exc
    return coordinates


def _validate_existing_manifest_compatibility(
    partial_dir: Path,
    immutable: Mapping[str, object],
) -> None:
    """Reject provenance drift before acquiring a mutating writer lock."""

    manifest_path = partial_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema_version") != PREPROCESSING_MANIFEST_SCHEMA_VERSION:
        raise CorpusRunError(
            "partial run has unsupported manifest schema_version"
        )
    for key, expected in immutable.items():
        if key not in manifest or manifest[key] != expected:
            raise CorpusRunError(
                f"partial run is incompatible at manifest field {key!r}"
            )


def _load_or_create_manifest(
    partial_dir: Path, immutable: Mapping[str, object]
) -> dict[str, object]:
    manifest_path = partial_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)
        if (
            manifest.get("schema_version")
            != PREPROCESSING_MANIFEST_SCHEMA_VERSION
        ):
            raise CorpusRunError(
                "partial run has unsupported manifest schema_version"
            )
        if not isinstance(manifest.get("complete"), bool):
            raise CorpusRunError(
                "partial run manifest complete must be boolean"
            )
        for key, expected in immutable.items():
            if key not in manifest or manifest[key] != expected:
                raise CorpusRunError(
                    f"partial run is incompatible at manifest field {key!r}"
                )
        return manifest
    if partial_dir.exists():
        raise CorpusRunError(
            f"partial run directory lacks a manifest: {partial_dir}"
        )
    manifest: dict[str, object] = {
        **immutable,
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
        "completed_row_groups": [],
        "relation_totals": {name: 0 for name in RELATION_NAMES},
        "outcome_totals": {},
        "complete": False,
    }
    partial_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = partial_dir.parent / f".{partial_dir.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_manifest(temporary, manifest)
        fsync_directory(temporary)
        os.replace(temporary, partial_dir)
        fsync_directory(partial_dir.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _publish_validated_complete_partial(
    *,
    partial_dir: Path,
    completed_dir: Path,
    manifest: Mapping[str, object],
    input_parquet: pq.ParquetFile,
    completed: set[int],
) -> None:
    if len(completed) != input_parquet.num_row_groups:
        raise CorpusRunError(
            "complete partial manifest does not cover every row group"
        )
    relation_paths = {
        relation: partial_dir / f"{relation}.parquet"
        for relation in RELATION_NAMES
    }
    totals = _validate_completed_relations(
        relation_paths=relation_paths,
        input_parquet=input_parquet,
        expected_rows=input_parquet.metadata.num_rows,
    )
    for key in ("relation_totals", "outcome_totals"):
        if manifest.get(key) != totals[key]:
            raise CorpusRunError(
                f"complete partial manifest field {key!r} is invalid"
            )
    hashes = manifest.get("relation_sha256")
    if not isinstance(hashes, dict) or any(
        not _is_sha256(hashes.get(relation))
        or hashes.get(relation) != file_sha256(path)
        for relation, path in relation_paths.items()
    ):
        raise CorpusRunError("complete partial relation hashes are invalid")
    for path in relation_paths.values():
        fsync_file(path)
    fsync_file(partial_dir / "manifest.json")
    fsync_directory(partial_dir)
    os.replace(partial_dir, completed_dir)
    fsync_directory(completed_dir.parent)


def _completed_row_groups(
    manifest: Mapping[str, object], *, expected_count: int
) -> set[int]:
    raw = manifest.get("completed_row_groups")
    if (
        not isinstance(raw, list)
        or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= expected_count
            for index in raw
        )
        or len(raw) != len(set(cast(list[int], raw)))
        or raw != sorted(raw)
    ):
        raise CorpusRunError(
            "manifest completed_row_groups must be sorted unique valid indexes"
        )
    return set(cast(list[int], raw))


def _validate_completed_parts(partial_dir: Path, completed: set[int]) -> None:
    for row_group_index in completed:
        part_id = _part_id(row_group_index)
        part_dir = partial_dir / "parts" / part_id
        try:
            manifest = read_part_manifest(part_dir)
        except ValueError as exc:
            raise CorpusRunError(str(exc)) from exc
        if manifest.get("schema_version") != PROJECTED_PART_SCHEMA_VERSION:
            raise CorpusRunError(f"part manifest schema mismatch: {part_id}")
        if manifest.get("part_id") != part_id:
            raise CorpusRunError(f"part manifest identity mismatch: {part_id}")
        relations = manifest.get("relations")
        if not isinstance(relations, dict) or set(relations) != set(
            RELATION_NAMES
        ):
            raise CorpusRunError(
                f"part manifest relations are incomplete: {part_id}"
            )
        for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
            path = part_dir / f"{relation}.parquet"
            metadata = relations.get(relation)
            if not isinstance(metadata, dict):
                raise CorpusRunError(
                    f"invalid {relation!r} checkpoint metadata: {part_id}"
                )
            try:
                parquet = pq.ParquetFile(path)
            except (OSError, pa.ArrowException) as exc:
                raise CorpusRunError(
                    f"invalid {relation!r} checkpoint: {part_id}"
                ) from exc
            if not parquet.schema_arrow.equals(schema):
                raise CorpusRunError(
                    f"checkpoint schema mismatch for {relation!r}: {part_id}"
                )
            if metadata.get("rows") != parquet.metadata.num_rows:
                raise CorpusRunError(
                    f"checkpoint row count mismatch for {relation!r}: {part_id}"
                )
            recorded_hash = metadata.get("sha256")
            if not _is_sha256(recorded_hash) or recorded_hash != file_sha256(
                path
            ):
                raise CorpusRunError(
                    f"checkpoint hash mismatch for {relation!r}: {part_id}"
                )


def _remove_orphan_part(partial_dir: Path, part_id: str) -> None:
    path = partial_dir / "parts" / part_id
    if path.exists():
        shutil.rmtree(path)


def _part_relation_totals(
    partial_dir: Path, completed: set[int]
) -> dict[str, int]:
    totals = {name: 0 for name in RELATION_NAMES}
    for index in completed:
        manifest = read_part_manifest(partial_dir / "parts" / _part_id(index))
        relations = cast(dict[str, dict[str, object]], manifest["relations"])
        for relation in RELATION_NAMES:
            totals[relation] += cast(int, relations[relation]["rows"])
    return totals


def _part_outcome_totals(
    partial_dir: Path, completed: set[int]
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for index in completed:
        parquet = pq.ParquetFile(
            partial_dir / "parts" / _part_id(index) / "results.parquet"
        )
        for batch in parquet.iter_batches(columns=["outcome"]):
            for value in batch.column("outcome").to_pylist():
                if not isinstance(value, str) or not value:
                    raise CorpusRunError(
                        "results must record a non-empty outcome"
                    )
                totals[value] += 1
    return dict(sorted(totals.items()))


def _validate_completed_relations(
    *,
    relation_paths: Mapping[str, Path],
    input_parquet: pq.ParquetFile,
    expected_rows: object,
) -> dict[str, object]:
    if set(relation_paths) != set(RELATION_NAMES):
        raise CorpusRunError("combined relations do not match the schema")
    if not isinstance(expected_rows, int) or expected_rows < 0:
        raise CorpusRunError("expected row count must be non-negative")
    totals: dict[str, int] = {}
    for relation, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        try:
            parquet = pq.ParquetFile(relation_paths[relation])
        except (OSError, pa.ArrowException) as exc:
            raise CorpusRunError(
                f"cannot read completed relation {relation!r}"
            ) from exc
        if not parquet.schema_arrow.equals(schema):
            raise CorpusRunError(
                f"completed relation has wrong schema: {relation!r}"
            )
        totals[relation] = parquet.metadata.num_rows
    if totals["results"] != expected_rows:
        raise CorpusRunError(
            "result row count does not match the input corpus row count"
        )
    outcomes = validate_preprocessing_relations(
        input_parquet=input_parquet,
        results_path=relation_paths["results"],
        candidates_path=relation_paths["candidates"],
        step_facts_path=relation_paths["step_facts"],
        rejections_path=relation_paths["rejections"],
    )
    if sum(outcomes.values()) != totals["results"]:
        raise CorpusRunError("outcome totals do not reconcile with results")
    return {
        "relation_totals": totals,
        "outcome_totals": dict(sorted(outcomes.items())),
    }


def validate_preprocessing_relations(
    *,
    input_parquet: pq.ParquetFile,
    results_path: Path,
    candidates_path: Path,
    step_facts_path: Path,
    rejections_path: Path,
) -> Counter[str]:
    input_rows = _rows(input_parquet, REQUIRED_INPUT_COLUMNS)
    results = _rows(pq.ParquetFile(results_path), tuple(RESULT_COLUMNS))
    candidates = _rows(
        pq.ParquetFile(candidates_path),
        tuple(PROJECTED_ARTIFACT_SCHEMAS["candidates"].names),
    )
    facts = _rows(
        pq.ParquetFile(step_facts_path),
        tuple(PROJECTED_ARTIFACT_SCHEMAS["step_facts"].names),
    )
    rejections = _rows(
        pq.ParquetFile(rejections_path),
        tuple(PROJECTED_ARTIFACT_SCHEMAS["rejections"].names),
    )
    current_result = next(results, None)
    current_candidate = next(candidates, None)
    current_fact = next(facts, None)
    current_rejection = next(rejections, None)
    outcomes: Counter[str] = Counter()
    with _sample_id_store() as seen:
        for source_row in input_rows:
            if current_result is None:
                raise CorpusRunError("results end before the input corpus")
            sample_id, expected_count, outcome = _validate_result(
                source_row, current_result
            )
            try:
                seen.execute(
                    "INSERT INTO sample_ids(sample_id) VALUES (?)",
                    (sample_id,),
                )
            except sqlite3.IntegrityError as exc:
                raise CorpusRunError(
                    "results contain duplicate sample_id values"
                ) from exc
            outcomes[outcome] += 1
            current_candidate, candidate_ids = _validate_candidates_for_sample(
                sample_id,
                expected_count,
                current_candidate,
                candidates,
            )
            current_fact, step_facts = _validate_facts_for_sample(
                sample_id, current_fact, facts
            )
            current_rejection = _validate_rejections_for_sample(
                sample_id,
                step_facts,
                current_rejection,
                rejections,
            )
            _validate_preprocessing_waterfall(
                decoder_output=source_row["decoder_output"],
                result=current_result,
                candidate_ids=candidate_ids,
                step_facts=step_facts,
            )
            current_result = next(results, None)
    if current_result is not None:
        raise CorpusRunError("results contain rows absent from input")
    for name, current in (
        ("candidates", current_candidate),
        ("step_facts", current_fact),
        ("rejections", current_rejection),
    ):
        if current is not None:
            raise CorpusRunError(
                f"{name} contains sample_id absent from results"
            )
    return outcomes


def validate_preprocessing_derivation(
    *,
    input_parquet: pq.ParquetFile,
    results_path: Path,
    candidates_path: Path,
    step_facts_path: Path,
    rejections_path: Path,
    preprocessing_config: PreprocessingConfig,
) -> None:
    """Replay the bound definition and require its exact canonical projection."""

    runner = bind_preprocessing(preprocessing_config)
    actual_rows = {
        "results": _rows(
            pq.ParquetFile(results_path),
            tuple(PROJECTED_ARTIFACT_SCHEMAS["results"].names),
        ),
        "candidates": _rows(
            pq.ParquetFile(candidates_path),
            tuple(PROJECTED_ARTIFACT_SCHEMAS["candidates"].names),
        ),
        "step_facts": _rows(
            pq.ParquetFile(step_facts_path),
            tuple(PROJECTED_ARTIFACT_SCHEMAS["step_facts"].names),
        ),
        "rejections": _rows(
            pq.ParquetFile(rejections_path),
            tuple(PROJECTED_ARTIFACT_SCHEMAS["rejections"].names),
        ),
    }
    for source_row in _rows(input_parquet, REQUIRED_INPUT_COLUMNS):
        sample_id = source_row["sample_id"]
        decoder_output = source_row["decoder_output"]
        if not isinstance(sample_id, str):
            raise CorpusRunError("sample_id must be a string")
        if decoder_output is not None and not isinstance(decoder_output, str):
            raise CorpusRunError("decoder_output must be a string or null")
        trace = (
            None
            if decoder_output is None
            else runner.run(TextArtifact(text=decoder_output))
        )
        expected = project_preprocessing_result(
            sample_id,
            decoder_output,
            trace,
        )
        for relation in RELATION_NAMES:
            expected_relation_rows = cast(
                list[dict[str, object]],
                getattr(expected, relation),
            )
            actual_relation_rows = actual_rows[relation]
            for expected_row in expected_relation_rows:
                actual_row = next(actual_relation_rows, None)
                if actual_row != expected_row:
                    raise CorpusRunError(
                        f"{relation} are not canonically derived from "
                        f"decoder_output for {sample_id!r}"
                    )
    for relation, rows in actual_rows.items():
        if next(rows, None) is not None:
            raise CorpusRunError(
                f"{relation} contain rows absent from the canonical replay"
            )


RESULT_COLUMNS: Final = (
    "sample_id",
    "decoder_output_presence",
    "raw_output_sha256",
    "outcome",
    "outcome_code",
    "failure_code",
    "failed_step",
    "cause",
    "propagated_through",
    "final_candidate_count",
)


def _validate_result(
    source: Mapping[str, object], result: Mapping[str, object]
) -> tuple[str, int, str]:
    sample_id = source["sample_id"]
    decoder_output = source["decoder_output"]
    if not isinstance(sample_id, str) or result["sample_id"] != sample_id:
        raise CorpusRunError("results sample_id does not match input")
    presence = "missing" if decoder_output is None else "present"
    if result["decoder_output_presence"] != presence:
        raise CorpusRunError(
            "results decoder_output_presence does not match input"
        )
    expected_hash = (
        None
        if decoder_output is None
        else _text_sha256(cast(str, decoder_output))
    )
    if result["raw_output_sha256"] != expected_hash:
        raise CorpusRunError("results raw_output_sha256 does not match input")
    count = result["final_candidate_count"]
    outcome = result["outcome"]
    if not isinstance(count, int) or count < 0:
        raise CorpusRunError("final_candidate_count must be non-negative")
    if not isinstance(outcome, str) or not outcome:
        raise CorpusRunError("outcome must be a non-empty string")
    if presence == "missing":
        expected = {
            "outcome": "decoder_output_missing",
            "outcome_code": None,
            "failure_code": None,
            "failed_step": None,
            "cause": None,
            "propagated_through": None,
            "final_candidate_count": 0,
        }
        if any(result[key] != value for key, value in expected.items()):
            raise CorpusRunError(
                "missing-output result has inconsistent fields"
            )
    elif count:
        if (
            result["outcome_code"] != outcome
            or result["failure_code"] is not None
            or result["failed_step"] is not None
            or result["cause"] is not None
            or result["propagated_through"] is not None
        ):
            raise CorpusRunError("successful result has inconsistent fields")
    elif (
        result["outcome_code"] is not None
        or result["failure_code"] != outcome
        or not isinstance(result["failed_step"], str)
        or not isinstance(result["cause"], str)
        or not isinstance(result["propagated_through"], list)
    ):
        raise CorpusRunError("failed result has inconsistent fields")
    return sample_id, count, outcome


def _validate_candidates_for_sample(
    sample_id: str,
    expected_count: int,
    current: dict[str, object] | None,
    rows: Iterator[dict[str, object]],
) -> tuple[dict[str, object] | None, tuple[str, ...]]:
    count = 0
    candidate_ids: list[str] = []
    while current is not None and current["sample_id"] == sample_id:
        index = current["candidate_index"]
        candidate_id = current["candidate_id"]
        source = current["cleaned_source"]
        source_hash = current["source_sha256"]
        if index != count:
            raise CorpusRunError(
                f"candidate indexes are not contiguous for {sample_id!r}"
            )
        if not isinstance(candidate_id, str) or not candidate_id:
            raise CorpusRunError("candidate_id must be non-empty")
        if candidate_id in candidate_ids:
            raise CorpusRunError(
                f"candidate_id is not unique within {sample_id!r}"
            )
        if not isinstance(source, str):
            raise CorpusRunError("cleaned_source must be a string")
        if source_hash != _text_sha256(source):
            raise CorpusRunError("candidate source_sha256 is invalid")
        if candidate_id != candidate_id_for_source(source):
            raise CorpusRunError("candidate_id is not content-derived")
        try:
            validate_origin_paths(current["origins"])
        except ValueError as exc:
            raise CorpusRunError(
                f"candidate provenance is invalid for {sample_id!r}"
            ) from exc
        candidate_ids.append(candidate_id)
        count += 1
        current = next(rows, None)
    if count != expected_count:
        raise CorpusRunError(
            f"candidate count does not match results for {sample_id!r}"
        )
    return current, tuple(candidate_ids)


def _validate_facts_for_sample(
    sample_id: str,
    current: dict[str, object] | None,
    rows: Iterator[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, _ParsedStepFact]]:
    parsed: dict[str, _ParsedStepFact] = {}
    previous_position = -1
    while current is not None and current["sample_id"] == sample_id:
        step_name = current["step_name"]
        if not isinstance(step_name, str) or not step_name:
            raise CorpusRunError("step fact name must be non-empty")
        if step_name in parsed:
            raise CorpusRunError(
                f"duplicate step fact for {sample_id!r}/{step_name!r}"
            )
        position = _FACT_POSITION.get(step_name)
        schemas = _FACT_SCHEMAS.get(step_name)
        if position is None or schemas is None:
            raise CorpusRunError(
                f"unknown step fact for {sample_id!r}/{step_name!r}"
            )
        if position <= previous_position:
            raise CorpusRunError(
                f"step facts are out of pipeline order for {sample_id!r}"
            )
        raw = _validate_canonical_object_json(
            current["facts_json"], label="facts_json"
        )
        value: FrozenModel | None = None
        errors: list[ValidationError] = []
        for schema in schemas:
            try:
                value = schema.model_validate_json(
                    cast(str, current["facts_json"]),
                    strict=True,
                )
            except ValidationError as exc:
                errors.append(exc)
                continue
            break
        if value is None:
            raise CorpusRunError(
                f"step facts do not match the typed schema for "
                f"{sample_id!r}/{step_name!r}: {errors[-1]}"
            )
        parsed[step_name] = _ParsedStepFact(raw=raw, value=value)
        previous_position = position
        current = next(rows, None)
    return current, parsed


def _validate_rejections_for_sample(
    sample_id: str,
    step_facts: Mapping[str, _ParsedStepFact],
    current: dict[str, object] | None,
    rows: Iterator[dict[str, object]],
) -> dict[str, object] | None:
    actual: list[dict[str, object]] = []
    while current is not None and current["sample_id"] == sample_id:
        step_name = current["step_name"]
        if step_name not in step_facts:
            raise CorpusRunError(
                f"rejection has no parent step fact for {sample_id!r}"
            )
        details = _validate_canonical_object_json(
            current["details_json"], label="details_json"
        )
        actual.append(
            {
                "step_name": step_name,
                "candidate_id": current["candidate_id"],
                "input_index": current["input_index"],
                "reason_code": current["reason_code"],
                "details": details,
            }
        )
        current = next(rows, None)
    expected: list[dict[str, object]] = []
    for step_name, parsed in step_facts.items():
        raw_rejections = parsed.raw.get("rejections", ())
        if not isinstance(raw_rejections, list):
            continue
        for rejection in raw_rejections:
            if not isinstance(rejection, dict):
                raise CorpusRunError("typed rejection is not an object")
            expected.append(
                {
                    "step_name": step_name,
                    "candidate_id": rejection.get("candidate_id"),
                    "input_index": rejection.get(
                        "input_index", rejection.get("index")
                    ),
                    "reason_code": rejection.get(
                        "reason_code", rejection.get("reason")
                    ),
                    "details": {
                        key: value
                        for key, value in rejection.items()
                        if key
                        not in {
                            "candidate_id",
                            "input_index",
                            "index",
                            "reason_code",
                            "reason",
                        }
                    },
                }
            )
    if actual != expected:
        raise CorpusRunError(
            f"rejection relation does not match step facts for {sample_id!r}"
        )
    return current


def _validate_preprocessing_waterfall(
    *,
    decoder_output: object,
    result: Mapping[str, object],
    candidate_ids: tuple[str, ...],
    step_facts: Mapping[str, _ParsedStepFact],
) -> None:
    if decoder_output is None:
        if step_facts or candidate_ids:
            raise CorpusRunError(
                "missing-output result must not have derived relations"
            )
        return
    if not isinstance(decoder_output, str):
        raise CorpusRunError("decoder_output must be a string or null")

    normalized = normalize_decoder_output(decoder_output)
    if not normalized.is_valid:
        validation = _require_fact(
            step_facts,
            "validate_decoder_output",
            _DecoderOutputValidationFacts,
        )
        expected_validation = _DecoderOutputValidationFacts(
            text_character_count=len(normalized.text),
            **normalized.facts,
        )
        if validation != expected_validation:
            raise CorpusRunError(
                "decoder-output validation facts do not match input"
            )
        if (
            result["failed_step"] != "validate_decoder_output"
            or result["failure_code"]
            != PreprocessingFailureCode.DECODER_OUTPUT_INVALID.value
            or result["outcome"]
            != PreprocessingFailureCode.DECODER_OUTPUT_INVALID.value
            or result["propagated_through"] != list(_PIPELINE_STEPS)
            or candidate_ids
            or tuple(step_facts) != ("validate_decoder_output",)
        ):
            raise CorpusRunError(
                "invalid decoder-output result does not match typed waterfall"
            )
        return
    if "validate_decoder_output" in step_facts:
        raise CorpusRunError(
            "valid decoder output has validation-failure facts"
        )

    require = _require_fact(
        step_facts, "require_nonblank_text", _RequireNonblankFacts
    )

    failure_step: str | None = (
        "require_nonblank_text" if not require.is_nonblank else None
    )
    current_count: int | None = None
    if failure_step is None:
        extracted = _require_fact(
            step_facts, "extract_candidates", _ExtractFacts
        )
        current_count = extracted.candidate_count
        if current_count == 0:
            failure_step = "extract_candidates"

    candidate_count_steps = (
        "strip_fences",
        "dedent",
        "normalize_smart_quotes",
        "split_on_name_guard",
        "expand_last_return_salvage",
        "repair_import_lines",
        "dedupe_imports",
        "filter_nonblank_candidates",
    )
    for step_name in candidate_count_steps:
        if failure_step is not None:
            break
        counts = _require_fact(
            step_facts,
            step_name,
            _CandidateCounts,
        )
        assert current_count is not None
        if counts.input_candidate_count != current_count:
            raise CorpusRunError(
                f"candidate count waterfall is inconsistent at {step_name!r}"
            )
        current_count = counts.output_candidate_count
        if step_name == "filter_nonblank_candidates" and current_count == 0:
            failure_step = step_name

    if failure_step is None:
        identified = _require_fact(
            step_facts,
            "identify_candidates",
            (_IdentifyFailureFacts, _IdentifyFacts),
        )
        assert current_count is not None
        if identified.input_candidate_count != current_count:
            raise CorpusRunError(
                "candidate count waterfall is inconsistent at "
                "'identify_candidates'"
            )
        if isinstance(identified, _IdentifyFailureFacts):
            failure_step = "identify_candidates"
            current_count = 0
            current_candidate_ids: tuple[str, ...] = ()
        else:
            current_count = identified.identified_candidate_count
            current_candidate_ids = tuple(
                item.candidate_id for item in identified.inspections
            )
    else:
        current_candidate_ids = ()

    filter_steps = (
        ("filter_plain_literal", _SimpleFilterFacts),
        ("filter_code_repr", _SimpleFilterFacts),
        ("filter_compilable", _DiagnosticFilterFacts),
        ("filter_has_top_level_function", _FunctionFilterFacts),
    )
    terminal_survivors: tuple[_FunctionDiagnostics, ...] = ()
    for step_name, schema in filter_steps:
        if failure_step is not None:
            break
        filtered = _require_fact(step_facts, step_name, schema)
        assert current_count is not None
        if filtered.input_candidate_count != current_count:
            raise CorpusRunError(
                f"candidate count waterfall is inconsistent at {step_name!r}"
            )
        if any(
            item.candidate_id != current_candidate_ids[item.input_index]
            for item in (*filtered.survivors, *filtered.rejections)
        ):
            raise CorpusRunError(
                f"candidate membership waterfall is inconsistent at "
                f"{step_name!r}"
            )
        current_count = filtered.survivor_candidate_count
        current_candidate_ids = tuple(
            item.candidate_id for item in filtered.survivors
        )
        if step_name == "filter_has_top_level_function":
            terminal_survivors = filtered.survivors
        if current_count == 0:
            failure_step = step_name

    if failure_step is None:
        materialized = _require_fact(
            step_facts, "materialize_candidates", _CandidateCount
        )
        assert current_count is not None
        if materialized.candidate_count != current_count:
            raise CorpusRunError(
                "candidate count waterfall is inconsistent at "
                "'materialize_candidates'"
            )
        returned = _require_fact(
            step_facts,
            "return_all",
            (_CandidateCount, _SuccessfulReturnFacts),
        )
        if returned.candidate_count != current_count:
            raise CorpusRunError(
                "candidate count waterfall is inconsistent at 'return_all'"
            )
        if isinstance(returned, _CandidateCount):
            failure_step = "return_all"

    if failure_step is None:
        if not isinstance(returned, _SuccessfulReturnFacts):
            raise CorpusRunError("successful preprocessing has no outcome")
        if result["outcome"] != returned.outcome_code:
            raise CorpusRunError(
                "preprocessing outcome does not match terminal step facts"
            )
        survivor_ids = tuple(item.candidate_id for item in terminal_survivors)
        if survivor_ids != candidate_ids:
            raise CorpusRunError(
                "final candidates do not match terminal survivor membership"
            )
        expected_steps = tuple(
            name for name in _FACT_SCHEMAS if name != "validate_decoder_output"
        )
    else:
        expected_failure = _FAILURE_BY_STEP.get(failure_step)
        if (
            expected_failure is None
            or result["failed_step"] != failure_step
            or result["failure_code"] != expected_failure.value
            or result["outcome"] != expected_failure.value
        ):
            raise CorpusRunError(
                "preprocessing failure does not match waterfall facts"
            )
        failure_position = _STEP_POSITION[failure_step]
        if result["propagated_through"] != list(
            _PIPELINE_STEPS[failure_position + 1 :]
        ):
            raise CorpusRunError(
                "preprocessing propagated steps do not match failed_step"
            )
        if candidate_ids:
            raise CorpusRunError(
                "failed preprocessing must not publish candidates"
            )
        expected_steps = tuple(
            name
            for name in _FACT_SCHEMAS
            if name != "validate_decoder_output"
            and _STEP_POSITION[name] <= failure_position
        )
    if tuple(step_facts) != expected_steps:
        raise CorpusRunError(
            "step fact membership does not match the preprocessing waterfall"
        )


def _require_fact(
    step_facts: Mapping[str, _ParsedStepFact],
    step_name: str,
    expected: type[FrozenModel] | tuple[type[FrozenModel], ...],
):
    parsed = step_facts.get(step_name)
    if parsed is None:
        raise CorpusRunError(
            f"preprocessing waterfall is missing facts for {step_name!r}"
        )
    if not isinstance(parsed.value, expected):
        raise CorpusRunError(
            f"preprocessing facts have the wrong terminal form for "
            f"{step_name!r}"
        )
    return parsed.value


def _validate_canonical_object_json(
    value: object, *, label: str
) -> dict[str, object]:
    if not isinstance(value, str):
        raise CorpusRunError(f"{label} must be a JSON string")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise CorpusRunError(f"{label} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise CorpusRunError(f"{label} must encode an object")
    if value != _canonical_json(decoded):
        raise CorpusRunError(f"{label} is not canonical JSON")
    return decoded


@contextmanager
def _sample_id_store() -> Iterator[sqlite3.Connection]:
    with tempfile.TemporaryDirectory(
        prefix="dr-code-corpus-validation-"
    ) as root:
        connection = sqlite3.connect(Path(root) / "sample-ids.sqlite3")
        try:
            connection.execute(
                "CREATE TABLE sample_ids(sample_id TEXT PRIMARY KEY)"
            )
            yield connection
        finally:
            connection.close()


def _rows(
    parquet: pq.ParquetFile, columns: tuple[str, ...]
) -> Iterator[dict[str, object]]:
    for batch in parquet.iter_batches(
        batch_size=_BATCH_ROW_LIMIT, columns=list(columns)
    ):
        values = [batch.column(name).to_pylist() for name in columns]
        for row in zip(*values, strict=True):
            yield dict(zip(columns, row, strict=True))


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusRunError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusRunError("manifest must contain a JSON object")
    return value


def _write_manifest(directory: Path, manifest: Mapping[str, object]) -> None:
    destination = directory / "manifest.json"
    temporary = directory / ".manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    fsync_directory(directory)


def _validate_run_output_namespace(
    *,
    root: Path,
    completed_dir: Path,
    partial_dir: Path,
    run_id: str,
) -> None:
    """Reject every run-owned symlink before a lock or artifact is created."""

    try:
        validate_reserved_path(
            root / f".{run_id}.lock",
            label="preprocessing writer lock",
        )
        validate_reserved_path(
            root / f".{partial_dir.name}.tmp",
            label="preprocessing staging directory",
        )
        validate_owned_tree(
            partial_dir,
            label="preprocessing partial run",
        )
        validate_owned_tree(
            completed_dir,
            label="preprocessing completed run",
        )
    except UnsafeOutputPathError as exc:
        raise CorpusRunError(str(exc)) from exc


def _source_coordinates() -> dict[str, object]:
    try:
        digest = package_source_digest()
    except (OSError, ValueError) as exc:
        raise CorpusRunError(
            "installed dr_code Python source evidence is unavailable"
        ) from exc
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CorpusRunError(
            "installed dr_code Python source evidence is invalid"
        )
    return {
        "dr_code_python_package_sha256": digest,
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
    }


def _validate_run_id(value: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or value.startswith(".")
        or value.endswith(".partial")
    ):
        raise CorpusRunError(
            "run_id must be a single non-empty public path segment and must "
            "not start with '.' or end with reserved suffix '.partial'"
        )


def _generated_run_id(input_coordinates: Mapping[str, object]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    input_prefix = str(input_coordinates["sha256"])[:8]
    definition = HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
    return (
        f"preprocessing-{timestamp}-{input_prefix}-"
        f"{definition.identity_hash()[:8]}"
    )


def _part_id(index: int) -> str:
    return f"row_group_{index:08d}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["CorpusRunError", "run_preprocessing_corpus"]
