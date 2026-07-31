"""Validated coordinates for immutable preprocessing artifacts.

This module sits below the viewer so analysis, comparison, and the interactive
service can share one artifact contract without creating a corpus-to-viewer
dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.candidate_evaluation import (
    MEMBERSHIP_SCHEMA,
    RESULTS_SCHEMA as EVALUATION_RESULTS_SCHEMA,
)
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_EVALUATION_COORDINATE_FIELDS,
    CANDIDATE_EVALUATION_MANIFEST_FIELDS,
    CANDIDATE_EVALUATION_SCHEMA_VERSION,
    CandidateEvaluationContractError,
    candidate_evaluation_identity,
    preprocessing_run_identity,
)
from dr_code.corpus.coordinate_validation import (
    CoordinateValidationError,
    validate_evaluation_coordinates,
    validate_preprocessing_coordinates,
)
from dr_code.corpus.evaluation_relations import (
    EvaluationRelationsError,
    validate_evaluation_relations,
)
from dr_code.corpus.evaluation_generation import (
    CURRENT_FILENAME,
    GENERATIONS_DIRECTORY,
    MANIFEST_FILENAME as EVALUATION_MANIFEST_FILENAME,
    MEMBERSHIP_FILENAME,
    POINTER_SCHEMA_VERSION,
    RESULTS_FILENAME as EVALUATION_RESULTS_FILENAME,
    EvaluationGeneration,
    EvaluationGenerationError,
    validate_captured_generation,
)
from dr_code.corpus.preprocessing_contract import (
    PREPROCESSING_INPUT_FIELDS,
    PREPROCESSING_MANIFEST_FIELDS,
    PREPROCESSING_MANIFEST_SCHEMA_VERSION,
)
from dr_code.corpus.preprocessing_artifacts import (
    PROJECTED_ARTIFACT_SCHEMAS,
    validate_origin_paths,
)
from dr_code.corpus.preprocessing_run import (
    CorpusRunError,
    validate_preprocessing_relations,
)
from dr_code.corpus.stable_files import StableFile, stable_file, stable_files
from dr_code.eval import PreprocessingConfig

PREPROCESSING_MANIFEST_FILENAME: Final = "manifest.json"
_DESCRIPTOR_FIELDS: Final = frozenset(
    {
        "label",
        "dataset_id",
        "corpus",
        "preprocessing",
        "candidate_evaluation",
    }
)
_SHA256_LENGTH: Final = 64
_EVALUATION_POINTER_FIELDS: Final = frozenset(
    {
        "schema_version",
        "generation_id",
        "manifest_sha256",
        "candidate_membership_sha256",
        "candidate_results_sha256",
    }
)
_PREPROCESSING_SOURCE_FIELDS: Final = frozenset(
    {
        "git_commit",
        "source_tree_sha256",
        "python_implementation",
        "python_version",
    }
)
_INSTALLED_ENVIRONMENT_FIELDS: Final = frozenset({"distributions", "identity"})
_DISTRIBUTION_FIELDS: Final = frozenset({"name", "version"})
_DATASET_FIELDS: Final = frozenset({"dataset_id", "split", "hf_revision"})
_REUSE_SOURCE_FIELDS: Final = frozenset(
    {
        "manifest_sha256",
        "candidate_membership_sha256",
        "candidate_results_sha256",
        "membership_rows",
        "result_rows",
    }
)
_REUSED_SOURCE_FIELDS: Final = frozenset(
    {*_REUSE_SOURCE_FIELDS, "reused_result_rows"}
)


class RunValidationError(ValueError):
    """A descriptor or one of its immutable artifacts is invalid."""


class _DuplicateJsonKeyError(ValueError):
    pass


class _NonFiniteJsonNumberError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    """Resolved, hash-authenticated coordinates for one viewer run."""

    run_id: str
    label: str
    dataset_id: str
    corpus_path: Path
    corpus_sha256: str
    preprocessing_manifest_path: Path
    preprocessing_manifest_sha256: str
    preprocessing_identity: str
    results_path: Path
    candidates_path: Path
    step_facts_path: Path
    rejections_path: Path
    artifact_sha256: Mapping[str, str]
    preprocessing_schema_version: int
    definition_id: str
    definition_version: str
    definition_identity: str
    evaluation_manifest_path: Path | None = None
    evaluation_root_path: Path | None = None
    evaluation_manifest_sha256: str | None = None
    evaluation_generation_id: str | None = None
    evaluation_pointer_sha256: str | None = None
    evaluation_identity: str | None = None
    candidate_membership_path: Path | None = None
    candidate_results_path: Path | None = None
    evaluation_coordinates: Mapping[str, object] | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        label: str,
        dataset_id: str,
        corpus_path: str | Path,
        preprocessing: str | Path,
        candidate_evaluation: str | Path | None = None,
    ) -> RunDescriptor:
        with _admit_run_descriptor(
            label=label,
            dataset_id=dataset_id,
            corpus_path=corpus_path,
            preprocessing=preprocessing,
            candidate_evaluation=candidate_evaluation,
        ) as (descriptor, captured):
            return _restore_source_paths(descriptor, captured)

    @classmethod
    def from_file(cls, path: str | Path) -> RunDescriptor:
        descriptor_path = _required_file(path, "run descriptor")
        try:
            with stable_file(
                descriptor_path, label="run descriptor"
            ) as snapshot:
                value = _json_object(snapshot.path, "run descriptor")
        except ValueError as exc:
            raise RunValidationError(str(exc)) from exc
        unknown = sorted(set(value).difference(_DESCRIPTOR_FIELDS))
        if unknown:
            raise RunValidationError(
                "run descriptor contains unknown field(s): "
                + ", ".join(unknown)
            )
        configured_label = value.get("label")
        if not isinstance(configured_label, str):
            raise RunValidationError("run descriptor requires string 'label'")
        configured_dataset_id = value.get("dataset_id")
        if not isinstance(configured_dataset_id, str):
            raise RunValidationError(
                "run descriptor requires string 'dataset_id'"
            )
        corpus = _descriptor_path(value, "corpus", required=True)
        preprocessing = _descriptor_path(value, "preprocessing", required=True)
        evaluation = _descriptor_path(
            value, "candidate_evaluation", required=False
        )
        return cls.from_paths(
            label=configured_label,
            dataset_id=configured_dataset_id,
            corpus_path=_relative_to(descriptor_path, cast(str, corpus)),
            preprocessing=_relative_to(
                descriptor_path, cast(str, preprocessing)
            ),
            candidate_evaluation=(
                _relative_to(descriptor_path, evaluation)
                if evaluation is not None
                else None
            ),
        )

    @property
    def has_evaluation(self) -> bool:
        return self.evaluation_manifest_path is not None

    def to_json(self) -> str:
        value = {
            field: str(item) if isinstance(item, Path) else thaw_json(item)
            for field in self.__dataclass_fields__
            if (item := getattr(self, field)) is not None
        }
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class _EvaluationSource:
    root: Path
    generation_id: str
    generation_dir: Path
    manifest_path: Path
    membership_path: Path
    results_path: Path
    pointer: dict[str, object]
    pointer_file: StableFile


@contextmanager
def admitted_run_descriptor(
    *,
    label: str,
    dataset_id: str,
    corpus_path: str | Path,
    preprocessing: str | Path,
    candidate_evaluation: str | Path | None = None,
) -> Iterator[RunDescriptor]:
    """Hold one validated disk-backed input snapshot for an operation."""

    with _admit_run_descriptor(
        label=label,
        dataset_id=dataset_id,
        corpus_path=corpus_path,
        preprocessing=preprocessing,
        candidate_evaluation=candidate_evaluation,
    ) as (descriptor, _captured):
        yield descriptor


@contextmanager
def _admit_run_descriptor(
    *,
    label: str,
    dataset_id: str,
    corpus_path: str | Path,
    preprocessing: str | Path,
    candidate_evaluation: str | Path | None,
) -> Iterator[tuple[RunDescriptor, Mapping[str, StableFile]]]:
    with ExitStack() as stack:
        try:
            admitted = _prepare_admitted_descriptor(
                stack=stack,
                label=label,
                dataset_id=dataset_id,
                corpus_path=corpus_path,
                preprocessing=preprocessing,
                candidate_evaluation=candidate_evaluation,
            )
        except RunValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise RunValidationError(str(exc)) from exc
        yield admitted


def _prepare_admitted_descriptor(
    *,
    stack: ExitStack,
    label: str,
    dataset_id: str,
    corpus_path: str | Path,
    preprocessing: str | Path,
    candidate_evaluation: str | Path | None,
) -> tuple[RunDescriptor, Mapping[str, StableFile]]:
    normalized_label = _nonblank(label, "run label")
    normalized_dataset_id = _exact_nonblank(dataset_id, "dataset_id")
    source_corpus = _required_file(corpus_path, "corpus")
    source_manifest = _resolve_manifest(
        preprocessing,
        PREPROCESSING_MANIFEST_FILENAME,
        "preprocessing",
    )
    source_root = source_manifest.parent
    capture_paths: dict[str, Path] = {
        "corpus": source_corpus,
        "preprocessing_manifest": source_manifest,
        **{
            f"preprocessing_{name}": _required_file(
                source_root / f"{name}.parquet", name
            )
            for name in PROJECTED_ARTIFACT_SCHEMAS
        },
    }
    evaluation_source: _EvaluationSource | None = None
    if candidate_evaluation is not None:
        evaluation_source = _evaluation_source(candidate_evaluation, stack)
        capture_paths.update(
            {
                "evaluation_manifest": evaluation_source.manifest_path,
                "candidate_membership": evaluation_source.membership_path,
                "candidate_results": evaluation_source.results_path,
            }
        )
    captured = stack.enter_context(stable_files(capture_paths))
    descriptor = _descriptor_from_captured(
        normalized_label=normalized_label,
        normalized_dataset_id=normalized_dataset_id,
        captured=captured,
        evaluation_source=evaluation_source,
    )
    return descriptor, captured


def _descriptor_from_captured(
    *,
    normalized_label: str,
    normalized_dataset_id: str,
    captured: Mapping[str, StableFile],
    evaluation_source: _EvaluationSource | None,
) -> RunDescriptor:
    corpus_file = captured["corpus"]
    manifest_file = captured["preprocessing_manifest"]
    manifest = _json_object(manifest_file.path, "preprocessing manifest")
    preprocessing_identity, preprocessing_config = (
        _validate_preprocessing_manifest(
            manifest,
            corpus_file.path,
            corpus_file.sha256,
            corpus_file.size,
        )
    )
    relation_files = {
        name: captured[f"preprocessing_{name}"]
        for name in PROJECTED_ARTIFACT_SCHEMAS
    }
    relation_paths = {
        name: stable.path for name, stable in relation_files.items()
    }
    hashes = _validate_preprocessing_relations(
        relation_paths,
        manifest,
        corpus_file.path,
        {name: stable.sha256 for name, stable in relation_files.items()},
    )

    evaluation_manifest_path: Path | None = None
    evaluation_manifest_sha256: str | None = None
    evaluation_identity: str | None = None
    membership_path: Path | None = None
    evaluation_results_path: Path | None = None
    evaluation_generation_id: str | None = None
    evaluation_pointer_sha256: str | None = None
    evaluation_coordinates: dict[str, object] | None = None
    if evaluation_source is not None:
        evaluation_manifest_file = captured["evaluation_manifest"]
        membership_file = captured["candidate_membership"]
        results_file = captured["candidate_results"]
        try:
            validate_captured_generation(
                EvaluationGeneration(
                    root=evaluation_source.root,
                    generation_id=evaluation_source.generation_id,
                    generation_dir=evaluation_source.generation_dir,
                    manifest_path=evaluation_source.manifest_path,
                    membership_path=evaluation_source.membership_path,
                    results_path=evaluation_source.results_path,
                    pointer=evaluation_source.pointer,
                ),
                manifest_sha256=evaluation_manifest_file.sha256,
                membership_sha256=membership_file.sha256,
                results_sha256=results_file.sha256,
            )
        except EvaluationGenerationError as exc:
            raise RunValidationError(str(exc)) from exc
        evaluation_manifest_path = evaluation_manifest_file.path
        membership_path = membership_file.path
        evaluation_results_path = results_file.path
        evaluation_generation_id = evaluation_source.generation_id
        evaluation_pointer_sha256 = evaluation_source.pointer_file.sha256
        evaluation_manifest = _json_object(
            evaluation_manifest_path, "candidate evaluation manifest"
        )
        evaluation_coordinates = _validate_evaluation(
            evaluation_manifest,
            corpus_path=corpus_file.path,
            candidates_path=relation_paths["candidates"],
            membership_path=membership_path,
            results_path=evaluation_results_path,
            corpus_sha256=corpus_file.sha256,
            preprocessing_identity=preprocessing_identity,
            preprocessing_manifest=manifest,
            preprocessing_config=preprocessing_config,
            membership_sha256=membership_file.sha256,
            results_sha256=results_file.sha256,
        )
        evaluation_identity = cast(
            str, evaluation_coordinates["evaluation_identity"]
        )
        authenticated_dataset_id = _evaluation_dataset_id(
            evaluation_coordinates
        )
        if authenticated_dataset_id != normalized_dataset_id:
            raise RunValidationError(
                "descriptor dataset_id does not match authenticated "
                "candidate evaluation dataset_id"
            )
        evaluation_manifest_sha256 = evaluation_manifest_file.sha256
        hashes["candidate_membership"] = membership_file.sha256
        hashes["candidate_results"] = results_file.sha256

    definition = preprocessing_config.definition_ref
    return RunDescriptor(
        run_id=cast(str, manifest["run_id"]),
        label=normalized_label,
        dataset_id=normalized_dataset_id,
        corpus_path=corpus_file.path,
        corpus_sha256=corpus_file.sha256,
        preprocessing_manifest_path=manifest_file.path,
        preprocessing_manifest_sha256=manifest_file.sha256,
        preprocessing_identity=preprocessing_identity,
        results_path=relation_paths["results"],
        candidates_path=relation_paths["candidates"],
        step_facts_path=relation_paths["step_facts"],
        rejections_path=relation_paths["rejections"],
        artifact_sha256=MappingProxyType(hashes),
        preprocessing_schema_version=PREPROCESSING_MANIFEST_SCHEMA_VERSION,
        definition_id=definition.definition_id,
        definition_version=definition.version,
        definition_identity=definition.identity_hash,
        evaluation_manifest_path=evaluation_manifest_path,
        evaluation_root_path=(
            evaluation_source.root if evaluation_source is not None else None
        ),
        evaluation_manifest_sha256=evaluation_manifest_sha256,
        evaluation_generation_id=evaluation_generation_id,
        evaluation_pointer_sha256=evaluation_pointer_sha256,
        evaluation_identity=evaluation_identity,
        candidate_membership_path=membership_path,
        candidate_results_path=evaluation_results_path,
        evaluation_coordinates=(
            cast(Mapping[str, object], _freeze_json(evaluation_coordinates))
            if evaluation_coordinates is not None
            else None
        ),
    )


def _evaluation_source(
    value: str | Path, stack: ExitStack
) -> _EvaluationSource:
    requested = Path(value).expanduser()
    if requested.is_symlink():
        raise RunValidationError("evaluation root must not be a symlink")
    try:
        root = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RunValidationError(
            f"evaluation root does not exist: {value}"
        ) from exc
    if not root.is_dir():
        raise RunValidationError(f"evaluation root is not a directory: {root}")
    pointer_path = root / CURRENT_FILENAME
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise RunValidationError(
            f"evaluation directory has no {CURRENT_FILENAME}: {root}"
        )
    pointer_file = stack.enter_context(
        stable_file(pointer_path, label="evaluation pointer")
    )
    pointer = _json_object(pointer_file.path, "evaluation pointer")
    if (
        set(pointer) != _EVALUATION_POINTER_FIELDS
        or pointer.get("schema_version") != POINTER_SCHEMA_VERSION
    ):
        raise RunValidationError(
            "evaluation pointer schema does not match schema_version 1"
        )
    generation_id = _sha256(
        pointer.get("generation_id"), "evaluation pointer generation_id"
    )
    if Path(generation_id).name != generation_id or generation_id.startswith(
        "."
    ):
        raise RunValidationError("evaluation pointer generation_id is invalid")
    generations = root / GENERATIONS_DIRECTORY
    if generations.is_symlink():
        raise RunValidationError(
            "evaluation generations directory must not be a symlink"
        )
    generation_dir = generations / generation_id
    if generation_dir.is_symlink() or not generation_dir.is_dir():
        raise RunValidationError(
            f"evaluation generation is missing or invalid: {generation_id}"
        )
    resolved_generation = generation_dir.resolve(strict=True)
    if resolved_generation.parent != generations.resolve(strict=True):
        raise RunValidationError(
            "evaluation generation escapes the generations directory"
        )
    paths = {
        EVALUATION_MANIFEST_FILENAME: (
            resolved_generation / EVALUATION_MANIFEST_FILENAME
        ),
        MEMBERSHIP_FILENAME: resolved_generation / MEMBERSHIP_FILENAME,
        EVALUATION_RESULTS_FILENAME: (
            resolved_generation / EVALUATION_RESULTS_FILENAME
        ),
    }
    if {path.name for path in resolved_generation.iterdir()} != set(paths):
        raise RunValidationError(
            "evaluation generation contains unexpected artifacts"
        )
    for filename, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise RunValidationError(
                f"evaluation generation artifact is missing: {filename}"
            )
    return _EvaluationSource(
        root=root,
        generation_id=generation_id,
        generation_dir=resolved_generation,
        manifest_path=paths[EVALUATION_MANIFEST_FILENAME],
        membership_path=paths[MEMBERSHIP_FILENAME],
        results_path=paths[EVALUATION_RESULTS_FILENAME],
        pointer=pointer,
        pointer_file=pointer_file,
    )


def _restore_source_paths(
    descriptor: RunDescriptor,
    captured: Mapping[str, StableFile],
) -> RunDescriptor:
    values: dict[str, object] = {
        "corpus_path": captured["corpus"].source_path,
        "preprocessing_manifest_path": captured[
            "preprocessing_manifest"
        ].source_path,
        **{
            f"{name}_path": captured[f"preprocessing_{name}"].source_path
            for name in PROJECTED_ARTIFACT_SCHEMAS
        },
    }
    if descriptor.has_evaluation:
        values.update(
            {
                "evaluation_manifest_path": captured[
                    "evaluation_manifest"
                ].source_path,
                "candidate_membership_path": captured[
                    "candidate_membership"
                ].source_path,
                "candidate_results_path": captured[
                    "candidate_results"
                ].source_path,
            }
        )
    return replace(descriptor, **values)


def normalize_origins(value: object) -> list[dict[str, object]]:
    """Parse the one current persisted origin representation."""
    return validate_origin_paths(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_preprocessing_manifest(
    manifest: dict[str, object],
    corpus: Path,
    corpus_sha256: str,
    corpus_size: int,
) -> tuple[str, PreprocessingConfig]:
    schema_version = _nonnegative_int(
        manifest.get("schema_version"),
        "preprocessing manifest schema_version",
    )
    if schema_version != PREPROCESSING_MANIFEST_SCHEMA_VERSION:
        raise RunValidationError(
            "preprocessing manifest requires schema_version "
            f"{PREPROCESSING_MANIFEST_SCHEMA_VERSION}"
        )
    if set(manifest) != PREPROCESSING_MANIFEST_FIELDS:
        raise RunValidationError(
            "preprocessing manifest fields do not match schema_version "
            f"{PREPROCESSING_MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("complete") is not True:
        raise RunValidationError("preprocessing manifest is incomplete")
    _nonblank(manifest.get("run_id"), "preprocessing run_id")
    input_value = manifest.get("input")
    if (
        not isinstance(input_value, dict)
        or set(input_value) != PREPROCESSING_INPUT_FIELDS
    ):
        raise RunValidationError("preprocessing manifest input is invalid")
    if input_value.get("sha256") != corpus_sha256:
        raise RunValidationError(
            "preprocessing manifest corpus fingerprint mismatch"
        )
    _nonblank(input_value.get("path"), "preprocessing input path")
    _sha256(input_value.get("sha256"), "preprocessing input sha256")
    _nonnegative_int(input_value.get("size"), "preprocessing input size")
    _nonnegative_int(
        input_value.get("expected_rows"),
        "preprocessing input expected_rows",
    )
    _nonnegative_int(
        input_value.get("expected_row_groups"),
        "preprocessing input expected_row_groups",
    )
    if not isinstance(input_value.get("schema_hex"), str):
        raise RunValidationError(
            "preprocessing input schema_hex must be a string"
        )
    corpus_parquet = _parquet(corpus, "corpus")
    required_fields = {
        "sample_id": pa.string(),
        "decoder_output": pa.string(),
    }
    for name, data_type in required_fields.items():
        index = corpus_parquet.schema_arrow.get_field_index(name)
        if index < 0:
            raise RunValidationError(
                f"corpus is missing required column {name!r}"
            )
        field = corpus_parquet.schema_arrow.field(index)
        if field.type != data_type or (name == "sample_id" and field.nullable):
            raise RunValidationError(
                f"corpus column {name!r} has an unexpected schema"
            )
    schema_hex = corpus_parquet.schema_arrow.serialize().to_pybytes().hex()
    expected_input = {
        "sha256": corpus_sha256,
        "size": corpus_size,
        "schema_hex": schema_hex,
        "expected_rows": corpus_parquet.metadata.num_rows,
        "expected_row_groups": corpus_parquet.num_row_groups,
        "row_groups": [
            {
                "index": index,
                "rows": corpus_parquet.metadata.row_group(index).num_rows,
                "total_byte_size": (
                    corpus_parquet.metadata.row_group(index).total_byte_size
                ),
            }
            for index in range(corpus_parquet.num_row_groups)
        ],
    }
    _validate_row_group_claims(input_value.get("row_groups"))
    mismatches = [
        field
        for field, expected_value in expected_input.items()
        if input_value.get(field) != expected_value
    ]
    if mismatches:
        raise RunValidationError(
            "preprocessing manifest corpus coordinate mismatch: "
            + ", ".join(mismatches)
        )
    completed_row_groups = manifest.get("completed_row_groups")
    if (
        not isinstance(completed_row_groups, list)
        or any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in completed_row_groups
        )
        or completed_row_groups != list(range(corpus_parquet.num_row_groups))
    ):
        raise RunValidationError(
            "preprocessing manifest completed_row_groups do not cover "
            "the captured corpus"
        )
    _positive_int(manifest.get("batch_size"), "preprocessing batch_size")
    started_at = _timestamp(manifest.get("started_at"), "started_at")
    updated_at = _timestamp(manifest.get("updated_at"), "updated_at")
    completed_at = _timestamp(manifest.get("completed_at"), "completed_at")
    if completed_at != updated_at or completed_at < started_at:
        raise RunValidationError(
            "preprocessing manifest timestamps are contradictory"
        )
    _validate_preprocessing_source(manifest.get("source"))
    _validate_installed_environment(manifest.get("installed_environment"))
    try:
        preprocessing_config = validate_preprocessing_coordinates(manifest)
        identity = preprocessing_run_identity(manifest)
    except (
        CandidateEvaluationContractError,
        CoordinateValidationError,
    ) as exc:
        raise RunValidationError(str(exc)) from exc
    return identity, preprocessing_config


def _validate_preprocessing_relations(
    paths: dict[str, Path],
    manifest: dict[str, object],
    corpus: Path,
    captured_hashes: Mapping[str, str],
) -> dict[str, str]:
    totals = manifest.get("relation_totals")
    recorded_hashes = manifest.get("relation_sha256")
    expected = set(PROJECTED_ARTIFACT_SCHEMAS)
    if not isinstance(totals, dict) or set(totals) != expected:
        raise RunValidationError(
            "preprocessing manifest relation totals are incomplete"
        )
    if (
        not isinstance(recorded_hashes, dict)
        or set(recorded_hashes) != expected
    ):
        raise RunValidationError(
            "preprocessing manifest relation hashes are incomplete"
        )
    validated_totals = cast(dict[str, object], totals)
    validated_hashes = cast(dict[str, object], recorded_hashes)
    hashes: dict[str, str] = {}
    input_value = cast(dict[str, object], manifest["input"])
    for name in expected:
        _nonnegative_int(
            validated_totals[name],
            f"preprocessing relation total {name}",
        )
        _sha256(
            validated_hashes[name],
            f"preprocessing relation hash {name}",
        )
    if validated_totals["results"] != input_value["expected_rows"]:
        raise RunValidationError(
            "preprocessing results do not cover every corpus row"
        )
    for name, schema in PROJECTED_ARTIFACT_SCHEMAS.items():
        parquet = _parquet(paths[name], name)
        if not parquet.schema_arrow.equals(schema):
            raise RunValidationError(
                f"{name}.parquet has an unexpected schema"
            )
        if validated_totals[name] != parquet.metadata.num_rows:
            raise RunValidationError(
                f"preprocessing manifest row count mismatch: {name}"
            )
        actual = captured_hashes[name]
        if validated_hashes[name] != actual:
            raise RunValidationError(
                f"preprocessing manifest hash mismatch: {name}"
            )
        hashes[name] = actual
    try:
        actual_outcomes = validate_preprocessing_relations(
            input_parquet=_parquet(corpus, "corpus"),
            results_path=paths["results"],
            candidates_path=paths["candidates"],
            step_facts_path=paths["step_facts"],
            rejections_path=paths["rejections"],
        )
    except CorpusRunError as exc:
        raise RunValidationError(
            f"preprocessing relations are contradictory: {exc}"
        ) from exc
    recorded_outcomes = _count_mapping(
        manifest.get("outcome_totals"),
        "preprocessing outcome_totals",
    )
    if recorded_outcomes != dict(sorted(actual_outcomes.items())):
        raise RunValidationError(
            "preprocessing manifest outcome_totals mismatch"
        )
    return hashes


def _validate_evaluation(
    manifest: dict[str, object],
    *,
    corpus_path: Path,
    candidates_path: Path,
    membership_path: Path,
    results_path: Path,
    corpus_sha256: str,
    preprocessing_identity: str,
    preprocessing_manifest: dict[str, object],
    preprocessing_config: PreprocessingConfig,
    membership_sha256: str,
    results_sha256: str,
) -> dict[str, object]:
    if set(manifest) != CANDIDATE_EVALUATION_MANIFEST_FIELDS:
        raise RunValidationError(
            "candidate evaluation manifest fields do not match schema_version "
            f"{CANDIDATE_EVALUATION_SCHEMA_VERSION}"
        )
    schema_version = _nonnegative_int(
        manifest.get("schema_version"),
        "candidate evaluation schema_version",
    )
    if schema_version != CANDIDATE_EVALUATION_SCHEMA_VERSION:
        raise RunValidationError(
            "candidate evaluation manifest requires schema_version "
            f"{CANDIDATE_EVALUATION_SCHEMA_VERSION}"
        )
    if manifest.get("complete") is not True:
        raise RunValidationError("candidate evaluation manifest is incomplete")
    preprocessing_run = manifest.get("preprocessing_run")
    if not isinstance(preprocessing_run, dict):
        raise RunValidationError(
            "candidate evaluation preprocessing coordinates are invalid"
        )
    relation_hashes = cast(
        dict[str, object], preprocessing_manifest["relation_sha256"]
    )
    relation_totals = cast(
        dict[str, object], preprocessing_manifest["relation_totals"]
    )
    expected_preprocessing = {
        "identity": preprocessing_identity,
        "relations": {
            name: {
                "sha256": relation_hashes[name],
                "rows": relation_totals[name],
            }
            for name in PROJECTED_ARTIFACT_SCHEMAS
        },
    }
    if (
        manifest.get("corpus_sha256") != corpus_sha256
        or preprocessing_run != expected_preprocessing
    ):
        raise RunValidationError(
            "candidate evaluation immutable input fingerprint mismatch"
        )
    membership = _parquet(membership_path, "candidate membership")
    results = _parquet(results_path, "candidate results")
    if not membership.schema_arrow.equals(MEMBERSHIP_SCHEMA):
        raise RunValidationError(
            "candidate_membership.parquet has an unexpected schema"
        )
    if not results.schema_arrow.equals(EVALUATION_RESULTS_SCHEMA):
        raise RunValidationError(
            "candidate_results.parquet has an unexpected schema"
        )
    checks = (
        (
            "candidate_membership_sha256",
            membership_sha256,
            "membership_rows",
            membership.metadata.num_rows,
        ),
        (
            "candidate_results_sha256",
            results_sha256,
            "result_rows",
            results.metadata.num_rows,
        ),
    )
    for hash_field, actual_hash, count_field, row_count in checks:
        _sha256(manifest.get(hash_field), hash_field)
        _nonnegative_int(manifest.get(count_field), count_field)
        if manifest.get(hash_field) != actual_hash:
            raise RunValidationError(
                f"candidate evaluation manifest {hash_field} mismatch"
            )
        if manifest.get(count_field) != row_count:
            raise RunValidationError(
                f"candidate evaluation manifest {count_field} mismatch"
            )
    _validate_evaluation_claim_types(manifest)
    coordinates = {
        field: manifest[field]
        for field in CANDIDATE_EVALUATION_COORDINATE_FIELDS
    }
    try:
        validate_evaluation_coordinates(
            coordinates,
            preprocessing_config=preprocessing_config,
        )
    except CoordinateValidationError as exc:
        raise RunValidationError(str(exc)) from exc
    recorded_evaluation_identity = _sha256(
        manifest.get("evaluation_identity"), "evaluation_identity"
    )
    try:
        expected_evaluation_identity = candidate_evaluation_identity(
            coordinates
        )
    except CandidateEvaluationContractError as exc:
        raise RunValidationError(str(exc)) from exc
    if recorded_evaluation_identity != expected_evaluation_identity:
        raise RunValidationError(
            "candidate evaluation evaluation_identity mismatch"
        )
    for field in (
        "snapshot_sha256",
        "metric_extraction_definition_identity",
        "metric_extraction_config_identity",
        "evaluation_procedure_definition_identity",
        "evaluation_procedure_config_identity",
        "runtime_identity",
    ):
        _sha256(coordinates[field], field)
    for field in (
        "runner_identity",
        "metrics_profile",
        "operator_name",
        "operator_version",
    ):
        _nonblank(coordinates[field], field)
    try:
        validate_evaluation_relations(
            corpus_path=corpus_path,
            candidates_path=candidates_path,
            membership_path=membership_path,
            results_path=results_path,
            coordinates=coordinates,
        )
    except EvaluationRelationsError as exc:
        raise RunValidationError(str(exc)) from exc
    _validate_evaluation_summaries(manifest, results_path=results_path)
    return {
        **coordinates,
        "evaluation_identity": recorded_evaluation_identity,
    }


def _evaluation_dataset_id(coordinates: dict[str, object]) -> str:
    dataset = coordinates.get("dataset")
    if not isinstance(dataset, dict):
        raise RunValidationError(
            "candidate evaluation dataset coordinates are invalid"
        )
    return _exact_nonblank(dataset.get("dataset_id"), "evaluation dataset_id")


def _required_file(value: str | Path, label: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RunValidationError(f"{label} does not exist: {value}") from exc
    if not path.is_file():
        raise RunValidationError(f"{label} is not a file: {path}")
    return path


def _resolve_manifest(value: str | Path, filename: str, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / filename
    return _required_file(path, f"{label} manifest")


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        _DuplicateJsonKeyError,
        _NonFiniteJsonNumberError,
    ) as exc:
        raise RunValidationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RunValidationError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise _NonFiniteJsonNumberError(f"non-finite JSON number: {value}")


def _descriptor_path(
    value: dict[str, object], field: str, *, required: bool
) -> str | None:
    item = value.get(field)
    if item is None and not required:
        return None
    if not isinstance(item, str) or not item.strip():
        raise RunValidationError(
            f"run descriptor requires nonblank string {field!r}"
        )
    return item


def _relative_to(descriptor: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else descriptor.parent / path


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunValidationError(f"{label} must be a nonblank string")
    return value.strip()


def _nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RunValidationError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise RunValidationError(f"{label} must be a positive integer")
    return result


def _timestamp(value: object, label: str) -> datetime:
    text = _nonblank(value, f"preprocessing {label}")
    try:
        result = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RunValidationError(
            f"preprocessing {label} must be an ISO-8601 timestamp"
        ) from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise RunValidationError(
            f"preprocessing {label} must include a UTC offset"
        )
    return result


def _validate_row_group_claims(value: object) -> None:
    if not isinstance(value, list):
        raise RunValidationError(
            "preprocessing input row_groups must be a list"
        )
    for position, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "index",
            "rows",
            "total_byte_size",
        }:
            raise RunValidationError(
                "preprocessing input row_groups contain invalid metadata"
            )
        index = _nonnegative_int(item.get("index"), "row group index")
        _nonnegative_int(item.get("rows"), "row group rows")
        _nonnegative_int(
            item.get("total_byte_size"), "row group total_byte_size"
        )
        if index != position:
            raise RunValidationError(
                "preprocessing input row_groups are not ordered"
            )


def _validate_preprocessing_source(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _PREPROCESSING_SOURCE_FIELDS
    ):
        raise RunValidationError(
            "preprocessing source coordinates are invalid"
        )
    _nonblank(value.get("git_commit"), "preprocessing source git_commit")
    _sha256(
        value.get("source_tree_sha256"),
        "preprocessing source source_tree_sha256",
    )
    _nonblank(
        value.get("python_implementation"),
        "preprocessing source python_implementation",
    )
    _nonblank(
        value.get("python_version"),
        "preprocessing source python_version",
    )


def _validate_installed_environment(value: object) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != _INSTALLED_ENVIRONMENT_FIELDS
    ):
        raise RunValidationError("installed_environment is invalid")
    distributions = value.get("distributions")
    if not isinstance(distributions, list):
        raise RunValidationError(
            "installed_environment distributions must be a list"
        )
    coordinates: list[dict[str, str]] = []
    for distribution in distributions:
        if (
            not isinstance(distribution, dict)
            or set(distribution) != _DISTRIBUTION_FIELDS
        ):
            raise RunValidationError(
                "installed_environment distribution is invalid"
            )
        coordinates.append(
            {
                "name": _nonblank(
                    distribution.get("name"), "distribution name"
                ),
                "version": _nonblank(
                    distribution.get("version"), "distribution version"
                ),
            }
        )
    if coordinates != sorted(coordinates, key=lambda item: item["name"]):
        raise RunValidationError(
            "installed_environment distributions are not ordered"
        )
    if len({item["name"] for item in coordinates}) != len(coordinates):
        raise RunValidationError(
            "installed_environment distributions contain duplicate names"
        )
    recorded_identity = _sha256(
        value.get("identity"), "installed_environment identity"
    )
    expected_identity = hashlib.sha256(
        json.dumps(
            coordinates,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recorded_identity != expected_identity:
        raise RunValidationError(
            "installed_environment identity does not match distributions"
        )


def _count_mapping(value: object, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise RunValidationError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        normalized_key = _nonblank(key, f"{label} key")
        if normalized_key != key:
            raise RunValidationError(f"{label} keys must be canonical")
        result[normalized_key] = _nonnegative_int(item, f"{label}[{key!r}]")
    return result


def _validate_evaluation_claim_types(manifest: Mapping[str, object]) -> None:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != _DATASET_FIELDS:
        raise RunValidationError("candidate evaluation dataset is invalid")
    for field in _DATASET_FIELDS:
        _nonblank(dataset.get(field), f"candidate evaluation dataset {field}")
    _nonnegative_int(
        manifest.get("max_infrastructure_retries"),
        "candidate evaluation max_infrastructure_retries",
    )
    _validate_installed_environment(manifest.get("installed_environment"))
    trusted_source = manifest.get("trusted_source_sha256")
    if not isinstance(trusted_source, dict) or not trusted_source:
        raise RunValidationError(
            "candidate evaluation trusted_source_sha256 is invalid"
        )
    for name, digest in trusted_source.items():
        _nonblank(name, "trusted source name")
        _sha256(digest, f"trusted source {name}")
    if not isinstance(manifest.get("host_runtime"), dict):
        raise RunValidationError(
            "candidate evaluation host_runtime must be an object"
        )


def _reuse_source(
    value: object,
    *,
    reused: bool,
    label: str,
) -> dict[str, object]:
    expected_fields = _REUSED_SOURCE_FIELDS if reused else _REUSE_SOURCE_FIELDS
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RunValidationError(f"{label} is invalid")
    result: dict[str, object] = {}
    for field in (
        "manifest_sha256",
        "candidate_membership_sha256",
        "candidate_results_sha256",
    ):
        result[field] = _sha256(value.get(field), f"{label} {field}")
    for field in ("membership_rows", "result_rows"):
        result[field] = _nonnegative_int(value.get(field), f"{label} {field}")
    if reused:
        result["reused_result_rows"] = _nonnegative_int(
            value.get("reused_result_rows"),
            f"{label} reused_result_rows",
        )
    return result


def _validate_evaluation_summaries(
    manifest: Mapping[str, object],
    *,
    results_path: Path,
) -> None:
    recorded_statuses = _count_mapping(
        manifest.get("record_status_totals"),
        "candidate evaluation record_status_totals",
    )
    actual_statuses: dict[str, int] = {}
    for batch in pq.ParquetFile(results_path).iter_batches(
        batch_size=65_536,
        columns=["record_status"],
    ):
        for status in batch.column("record_status").to_pylist():
            key = _nonblank(status, "candidate result record_status")
            actual_statuses[key] = actual_statuses.get(key, 0) + 1
    actual_statuses = dict(sorted(actual_statuses.items()))
    if recorded_statuses != actual_statuses:
        raise RunValidationError(
            "candidate evaluation record_status_totals mismatch"
        )

    raw_sources = manifest.get("reuse_result_sources")
    raw_reused_sources = manifest.get("reused_result_rows_by_source")
    if not isinstance(raw_sources, list) or not isinstance(
        raw_reused_sources, list
    ):
        raise RunValidationError(
            "candidate evaluation reuse summaries must be lists"
        )
    sources = [
        _reuse_source(
            value,
            reused=False,
            label=f"candidate evaluation reuse source {index}",
        )
        for index, value in enumerate(raw_sources)
    ]
    reused_sources = [
        _reuse_source(
            value,
            reused=True,
            label=f"candidate evaluation reused source {index}",
        )
        for index, value in enumerate(raw_reused_sources)
    ]
    identities = [
        (
            source["manifest_sha256"],
            source["candidate_membership_sha256"],
            source["candidate_results_sha256"],
        )
        for source in sources
    ]
    if len(identities) != len(set(identities)):
        raise RunValidationError(
            "candidate evaluation reuse sources contain duplicates"
        )
    stripped_reused_sources = [
        {field: source[field] for field in _REUSE_SOURCE_FIELDS}
        for source in reused_sources
    ]
    if sources != stripped_reused_sources:
        raise RunValidationError(
            "candidate evaluation reuse source summaries mismatch"
        )
    for source in reused_sources:
        if cast(int, source["reused_result_rows"]) > cast(
            int, source["result_rows"]
        ):
            raise RunValidationError(
                "candidate evaluation reused source row count is invalid"
            )
    total_reused = _nonnegative_int(
        manifest.get("reused_result_rows"),
        "candidate evaluation reused_result_rows",
    )
    if total_reused != sum(
        cast(int, source["reused_result_rows"]) for source in reused_sources
    ):
        raise RunValidationError(
            "candidate evaluation reused_result_rows mismatch"
        )
    result_rows = _nonnegative_int(
        manifest.get("result_rows"),
        "candidate evaluation result_rows",
    )
    if total_reused > result_rows:
        raise RunValidationError(
            "candidate evaluation reused_result_rows exceeds result_rows"
        )


def _exact_nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise RunValidationError(
            f"{label} must be a nonblank string without surrounding whitespace"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _parquet(path: Path, label: str) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise RunValidationError(
            f"{label} is not a readable Parquet file"
        ) from exc


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def thaw_json(value: object) -> object:
    """Return mutable JSON containers for a recursively frozen coordinate."""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


__all__ = (
    "PREPROCESSING_MANIFEST_FILENAME",
    "RunDescriptor",
    "RunValidationError",
    "admitted_run_descriptor",
    "file_sha256",
    "normalize_origins",
    "thaw_json",
)
